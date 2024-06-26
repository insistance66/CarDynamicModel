import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit

class CarDynamicModel:

    def __init__(self, DB, fs=100, time=20, flag=1, di=0, method='linear', solve_method='rk4'):
        """
        初始化车辆动态模型

        参数:
        DB : DB_data
            数据库类实例，包含车辆数据
        fs : int, optional, default=100
            采样频率
        time : int, optional, default=20
            模拟时间
        flag : int, optional, default=1
            标志变量，控制 e1 和 e2 的初值
        di : int, optional, default=0
            索引偏移量
        method : str, optional, default='linear'
            插值方法
        solve_method : str, optional, default='rk4'
            求解方法，'rk4', 'odeint', 'implicit_rk4'

        非线性求解方法中可换选项：（默认为Radau）
        Radau: 隐式 Radau IIA 方法，适用于刚性问题和高精度要求。
        BDF: 后向微分公式法，也适用于刚性问题。
        LSODA: 自动选择 Adams 方法或 BDF 方法，适用于非刚性和刚性问题。
            
        """
        self.DB = DB
        self.flag = flag
        self.method = method
        self.solve_method = solve_method

        self.RE1 = int(DB.RE1)
        if DB.RE2 == 'F23_A20':
            self.RE2 = 1
        elif DB.RE2 == 'F32_A30':
            self.RE2 = 2
        elif DB.RE2 == 'F23_A30':
            self.RE2 = 3
        elif DB.RE2 == 'F32_A20':
            self.RE2 = 4

        self.variables = ['k1', 'k2', 'k3', 'k4', 'Fz1', 'Fz2', 'Fz3', 'Fz4', 'Fy1', 'Fy2', 'Fy3', 'Fy4', 'Fs', \
                          'Mz', 'delta', 'e1', 'e1_1', 'e1_2', 'e2', 'e2_1', 'e2_2', 'vx', 'vy', 'a1', 'a2', 'a3', 'a4', \
                          'driver_torque', 'assist_torque', 'Total_torque', 'feedback', 'steering']
        self.index = DB.wst1 + di
        self.fs = fs
        self.time = time
        self.basic_information()
        self.coff()
        self.n = self.t1.shape[0]
        self.resample(self.fs)
        self.cut_by_time()
        self.F_hat()

    def basic_information(self):
        """
        提取车辆基本信息
        """
        self.Mv = np.unique(self.DB.data['Total weight'])  # kg
        self.Ix = np.unique(self.DB.data['Ixx'])           # kg.m^2
        self.Iy = np.unique(self.DB.data['Iyy'])           # kg.m^2
        self.Iz = np.unique(self.DB.data['Izz'])           # kg.m^2
        self.L1 = 1.02
        self.L2 = 2.64 - self.L1

    def coff(self):
        self.t = self.DB.data['Time']
        self.t1 = self.DB.data['Time'][self.index:]
        self.lp = self.DB.data['Lane gap']

        # 竖向接触力
        Fz1 = self.DB.data_table['Force/Axle 0/Left/Z'][1:].to_numpy().astype(float)
        Fz2 = self.DB.data_table['Force/Axle 1/Left/Z'][1:].to_numpy().astype(float)
        Fz3 = self.DB.data_table['Force/Axle 0/Right/Z'][1:].to_numpy().astype(float)
        Fz4 = self.DB.data_table['Force/Axle 1/Right/Z'][1:].to_numpy().astype(float)
        self.Fz1 = Fz1[self.index:] * 10     # N
        self.Fz2 = Fz2[self.index:] * 10     # N
        self.Fz3 = Fz3[self.index:] * 10     # N
        self.Fz4 = Fz4[self.index:] * 10     # N

        # 车轮侧向力
        Fy1 = self.DB.data_table['Force/Axle 0/Left/Y'][1:].to_numpy().astype(float)
        Fy2 = self.DB.data_table['Force/Axle 1/Left/Y'][1:].to_numpy().astype(float)
        Fy3 = self.DB.data_table['Force/Axle 0/Right/Y'][1:].to_numpy().astype(float)
        Fy4 = self.DB.data_table['Force/Axle 1/Right/Y'][1:].to_numpy().astype(float)
        self.Fy1 = Fy1[self.index:] * 10     # N
        self.Fy2 = Fy2[self.index:] * 10     # N
        self.Fy3 = Fy3[self.index:] * 10     # N
        self.Fy4 = Fy4[self.index:] * 10     # N

        # 侧滑刚度
        k1 = self.DB.data['Cornering stiffness/Axle 0/Left']  # N/rad
        k2 = self.DB.data['Cornering stiffness/Axle 1/Left']
        k3 = self.DB.data['Cornering stiffness/Axle 0/Right']
        k4 = self.DB.data['Cornering stiffness/Axle 1/Right']
        self.k1 = k1[self.index:]     # rad
        self.k2 = k2[self.index:]     # rad
        self.k3 = k3[self.index:]     # rad
        self.k4 = k4[self.index:]     # rad

        # 车轮侧滑角
        self.a1 = self.DB.data['Sideslip angle/Axle 0/Left'][self.index:]
        self.a2 = self.DB.data['Sideslip angle/Axle 1/Left'][self.index:]
        self.a3 = self.DB.data['Sideslip angle/Axle 0/Right'][self.index:]
        self.a4 = self.DB.data['Sideslip angle/Axle 1/Right'][self.index:]

        # 方向盘转角 相关的力
        self.driver_torque = self.DB.data['Steering wheel torque applied by pilot'][self.index:]
        self.assist_torque = self.DB.data['Steering assistance torque'][self.index:]
        self.Total_torque = self.DB.data['Total Torque at steering rack'][self.index:]
        self.feedback = self.DB.data['Model steering wheel torque feedback'][self.index:]
        self.steering = self.DB.steering_angle[self.index:] * np.pi/180

        # 车速
        self.vx = self.DB.data['Speed/X'][self.index:] / 3.6
        self.vx[self.vx == 0] = 1e-5
        self.vy = self.DB.data['Speed/Y'][self.index:] / 3.6

        # 侧向力
        self.Fs = self.DB.data['Side force - FCs'][self.index:]
        # 风引起的扭矩
        U = self.DB.data['Aerodynamic air speed']
        Cr2 = self.DB.data['Cr']
        rho = 1.225
        Mz = 1/2 * rho * U**2 * Cr2
        self.Mz = Mz[self.index:]

        # 横向误差
        if self.flag == 1:
            self.e1 = self.lp[self.index:] - self.lp[self.index]
    
            self.yaw = self.DB.yaw * np.pi/180
            self.e2 = self.yaw[self.index:] - self.yaw[self.index]
        else:
            self.e1 = self.lp[self.index:]
    
            self.yaw = self.DB.yaw * np.pi/180
            self.e2 = self.yaw[self.index:]

        e1_1 = self.vy + self.vx * self.yaw[self.index:]
        self.e1_1 = e1_1

        e2_1 = self.DB.data['Yaw speed'][self.index:] * np.pi/180
        self.e2_1 = e2_1

        ya = np.gradient(np.squeeze(self.vy), np.squeeze(self.t1))
        yaw1 = self.DB.data['Yaw speed'][self.index:] * np.pi/180
        self.e1_2 = ya + yaw1 * self.vx

        yaw2 = self.DB.data['Yaw acceleration'] * np.pi/180
        self.e2_2 = yaw2[self.index:]

        dalta = self.DB.steering1 * np.pi/180
        self.delta = dalta[self.index:]

        self.alpha_f = (self.vy + self.e2_1*self.L1)/self.vx - self.delta
        self.alpha_r = (self.vy - self.e2_1*self.L2)/self.vx

    def resample(self, fs=100):
        """
        重新采样数据
        
        参数:
        fs : int, optional, default=100
            采样频率
        """
        def interpolate_variable(variable, t1, t2, method=self.method):
            f = interp1d(t1, variable, kind=method)
            return f(t2)

        n1 = int((self.t1[-1] - self.t1[0]) * fs)
        self.t2 = np.linspace(self.t1[0], self.t1[-1], n1)

        for var in self.variables:
            setattr(self, var, interpolate_variable(getattr(self, var), self.t1, self.t2))

        self.t2 = self.t2 - self.t2[0]

    def cut_by_time(self):
        """
        根据时间截取数据
        """
        for var in self.variables:
            setattr(self, var, getattr(self, var)[:self.time * self.fs])

        self.t = self.t2[:self.time * self.fs]

        self.cf = -(self.k1 + self.k3)
        self.cr = -(self.k2 + self.k4)

        self.Ms = self.Mz - self.Fs/2 *(self.L1 - self.L2)

    def F_hat(self):
        self.F_hat = self.cf * self.alpha_f + self.cr * self.alpha_r - self.Mv * self.e1_2
        self.M_hat = self.Iz * self.e2_2 - (self.cf * self.alpha_f * self.L1 - self.cr * self.alpha_r * self.L2)

    def Recal(self):
        Mv = car.Mv
        Iz = car.Iz
        L1 = car.L1
        L2 = car.L2
        cf = car.cf
        cr = car.cr
        vx = car.vx
        vy = car.vy
        self.F1 = (cf+cr)/vx * car.e1_1 - (cf+cr) * car.e2 + (L1*cf-L2*cr)/vx * car.e2_1 \
                    - cf*car.delta - Mv*car.e1_2
        M = (L1*cf-L2*cr)/vx * car.e1_1 - (L1*cf-L2*cr) * car.e2 + (L1*L1*cf+L2*L2*cr)/vx * car.e2_1 \
                    - L1*cf*car.delta - Iz*car.e2_2
        self.M1 = -M

    
    @staticmethod
    def lateral_equation(t, X, cf, cr, lf, lr, v, Mv, Iz, delta, Fy, Mz):
        """
        计算车辆侧向运动方程

        参数:
        X : array
            当前状态变量
        t : float
            当前时间
        cf, cr : function
            前后侧滑刚度插值函数
        lf, lr : float
            前后轴距
        v : function
            车速插值函数
        Mv : float
            车辆质量
        Iz : float
            车辆绕z轴的转动惯量
        delta : function
            方向盘转角插值函数
        Fy : function
            侧向力插值函数
        Mz : function
            风引起的扭矩插值函数

        返回:
        array
            状态变量的导数
        """
        cf_t = cf(t)
        cr_t = cr(t)
        v_t = v(t)
        delta_t = delta(t)
        Fy_t = Fy(t)
        Mz_t = Mz(t)

        e1 = (cf_t + cr_t) / (Mv * v_t)
        e2 = -(cf_t + cr_t) / Mv
        e3 = (cf_t * lf - cr_t * lr) / (Mv * v_t)
        f1 = (cf_t * lf - cr_t * lr) / (Iz * v_t)
        f2 = -(cf_t * lf - cr_t * lr) / Iz
        f3 = (cf_t * lf ** 2 + cr_t * lr ** 2) / (Iz * v_t)
        e1, e2, e3, f1, f2, f3 = e1.item(), e2.item(), e3.item(), f1.item(), f2.item(), f3.item()

        A = np.array([
            [0, 1, 0, 0],
            [0, e1, e2, e3],
            [0, 0, 0, 1],
            [0, f1, f2, f3]
        ])

        e1 = -cf_t / Mv
        e2 = -cf_t * lf / Iz
        e1, e2 = e1.item(), e2.item()

        B = np.array([
            [0],
            [e1],
            [0],
            [e2]
        ])

        e1 = -Fy_t / Mv
        e2 = Mz_t / Iz
        e1, e2 = e1.item(), e2.item()

        C = np.array([
            [0],
            [e1],
            [0],
            [e2]
        ])

        X = X.reshape((-1, 1))
        X1 = A @ X + B * delta_t + C

        return X1.flatten()

    def rk4_step(self, func, y, t, dt, *args):
        """
        显式RK4方法一步求解

        参数:
        func : function
            动力学方程
        y : array
            当前状态变量
        t : float
            当前时间
        dt : float
            时间步长
        args : tuple
            额外参数

        返回:
        array
            更新后的状态变量
        """
        k1 = dt * func(t, y, *args)
        k2 = dt * func(t + dt / 2, y + k1 / 2, *args)
        k3 = dt * func(t + dt / 2, y + k2 / 2, *args)
        k4 = dt * func(t + dt, y + k3, *args)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve_dynamics(self, X0=[0, 0, 0, 0]):
        """
        解决车辆动力学问题

        参数:
        X0 : list, optional, default=[0, 0, 0, 0]
            初始状态变量

        返回:
        tuple
            状态变量 (X) 和 导数 (X1) 的数组
        """
        t = self.t
        dt = t[1] - t[0]
        n = len(t)
        X = np.zeros((n, len(X0)))
        X1 = np.zeros((n, len(X0)))
        X[0] = X0

        cf, cr, lf, lr, v, delta, Fy, Mz, Mv, Iz \
        = self.cf, self.cr, self.L1, self.L2, self.vx, self.delta, self.Fs, self.Ms, self.Mv, self.Iz

        cf_interp = interp1d(t, cf, kind='linear', fill_value="extrapolate")
        cr_interp = interp1d(t, cr, kind='linear', fill_value="extrapolate")
        v_interp = interp1d(t, v, kind='linear', fill_value="extrapolate")
        delta_interp = interp1d(t, delta, kind='linear', fill_value="extrapolate")
        Fy_interp = interp1d(t, Fy, kind='linear', fill_value="extrapolate")
        Mz_interp = interp1d(t, Mz, kind='linear', fill_value="extrapolate")

        if self.solve_method == 'rk4':
            for i in range(1, n):
                args = (cf_interp, cr_interp, lf, lr, v_interp, Mv, Iz, delta_interp, Fy_interp, Mz_interp)
                X[i] = self.rk4_step(self.lateral_equation, X[i-1], t[i-1], dt, *args)
                X1[i] = self.lateral_equation(t[i], X[i], *args)
        elif self.solve_method == 'odeint':
            args = (cf_interp, cr_interp, lf, lr, v_interp, Mv, Iz, delta_interp, Fy_interp, Mz_interp)
            X = odeint(self.lateral_equation, X0, t, args=args)
            for i in range(n):
                X1[i] = self.lateral_equation(t[i], X[i], cf_interp, cr_interp, lf, lr, v_interp, Mv, Iz, delta_interp, Fy_interp, Mz_interp)
        elif self.solve_method == 'implicit_rk4':
            args = (cf_interp, cr_interp, lf, lr, v_interp, Mv, Iz, delta_interp, Fy_interp, Mz_interp)
            sol = solve_ivp(self.lateral_equation, [t[0], t[-1]], X0, t_eval=t, method='Radau', args=args)
            X = sol.y.T
            for i in range(n):
                X1[i] = self.lateral_equation(t[i], X[i], cf_interp, cr_interp, lf, lr, v_interp, Mv, Iz, delta_interp, Fy_interp, Mz_interp)
        else:
            raise ValueError(f"Unknown solve method: {self.solve_method}")

        return X, X1

    def cubic_detrend(self, t, e1, y, flag=0, title='e1'):
        """
        使用三次模型去趋势

        参数:
        t : array
            时间数组
        e1 : array
            原始数据
        y : array
            计算数据
        flag : int, optional, default=0
            标志变量，控制是否输出图片
        title : str, optional, default='e1'
            图像标题

        返回:
        tuple
            去趋势后的数据、修正后的数据、残差和拟合偏差
        """
        dict1 = {'e1': 'y', 'e1_1': 'dy', 'e2': 's', 'e2_1': 'ds'}
        unit = {'e1': 'm', 'e1_1': 'm/s', 'e2': 'rad', 'e2_1': 'rad/s'}
        residuals = e1 - y

        def cubic_model(t, a, b, c):
            return a * t**3 + b * t**2 + c * t

        params, _ = curve_fit(cubic_model, t, residuals)
        a, b, c = params

        fitted_bias = cubic_model(t, a, b, c)
        corrected_y = y + fitted_bias

        if flag == 1:
            t1 = title
            t2 = dict1[t1]
            
            plt.figure(figsize=(14, 13))

            plt.subplot(3, 1, 1)
            plt.plot(t, e1, label=f'Original {t1}')
            plt.plot(t, y, label=f'Computed {t2}')
            plt.ylabel(unit[t1])
            plt.legend()
            plt.title(f'Original {t1} and Computed {t2}')

            plt.subplot(3, 1, 2)
            plt.plot(t, e1, label=f'Original {t1}')
            plt.plot(t, corrected_y, label=f'Corrected {t2}', linestyle='--')
            plt.ylabel(unit[t1])
            plt.legend()
            plt.title(f'Original {t1} and Corrected {t2}')

            plt.subplot(3, 1, 3)
            plt.plot(t, residuals, label='Residuals')
            plt.plot(t, fitted_bias, label='Fitted Bias', linestyle='--')
            plt.xlabel('s')
            plt.ylabel(f'Residual Value ({unit[t1]})')
            plt.legend()
            plt.title('Residuals and Fitted Bias')

            folder_path = f'系统误差2/{title}'
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            file = f'系统误差2/{title}/{self.DB.RE1}_{self.DB.RE2}.jpg'
            plt.savefig(file, format='jpg', bbox_inches='tight', pad_inches=0, dpi=600, transparent=True)

        data = [self.RE1, self.RE2, a, b, c]

        return data, corrected_y, residuals, fitted_bias

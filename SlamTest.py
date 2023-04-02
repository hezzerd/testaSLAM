import numpy as np
from roboticstoolbox import EKF, models, Sensor, uncertainties

# Загрузка логов и извлечение данных одометрии и измерения лидара
logs = []
for i in range(2, 17):
    with open(f"slam/examp{i}.txt") as f:
        logs.append(f.readlines())

odom_data = []
lidar_data = []
for log in logs:
    for line in log:
        data = line.split(";")
        odom = data[0].split(",")
        x, y, theta = map(float, odom[:3])
        odom_data.append([x, y, theta])
        lidar = data[1].split(",")
        lidar = np.array(list(map(float, lidar)))
        lidar_data.append(lidar)

# Фильтрация измерений лидара в диапазоне от 0.7 до 5.6
lidar_data_filtered = []
for lidar in lidar_data:
    mask = (lidar > 0.7) & (lidar < 5.6)
    lidar_data_filtered.append(lidar[mask])

# Определение размеров карты и ячеек
map_size = [10, 10]
cell_size = 0.1

# Инициализация EKF для SLAM
ekf = EKF(models.Pose2D(), Q=0.01)

# Инициализация датчика лидара
sensor = Sensor("lidar", uncertainties=[0.05, np.deg2rad(0.5)])

# Обработка данных одометрии и измерения лидара для SLAM
for i in range(len(odom_data)):
    # Обновление состояния EKF на основе данных одометрии
    ekf.predict(u=odom_data[i])
    
    # Обновление состояния EKF на основе данных лидара
    z = sensor.lidar(lidar_data_filtered[i])
    ekf.update(z, sensor)
    
    # Отображение карты окружающей среды
    ekf.plot_map(map_size=map_size, cell_size=cell_size)

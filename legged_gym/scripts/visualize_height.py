import numpy as np
import matplotlib.pyplot as plt

# 加载保存的数据
data = np.load('logs/height_history_20250417_165302_env0.npy')

# 绘制高度随时间变化图
plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['height'])
plt.xlabel('Time (s)')
plt.ylabel('Base Height (m)')
plt.title('Robot Base Height over Time')
plt.grid(True)
plt.savefig('height_plot.png')
plt.show()

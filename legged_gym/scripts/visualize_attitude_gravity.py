import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 获取文件路径，如果没有提供则使用默认值
if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    # 假设文件在logs/attitude_history目录下
    data_path = os.path.join("logs", "attitude_history", "attitude_history_20250419_114319_env0.npy")

# 检查文件是否存在
if not os.path.exists(data_path):
    print(f"Error: File {data_path} does not exist")
    sys.exit(1)

# 加载数据
try:
    attitude_data = np.load(data_path)
    print(f"Successfully loaded data file: {data_path}")
    print(f"Number of data points: {len(attitude_data)}")
    print(f"Data fields: {attitude_data.dtype.names}")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# 弧度
pitch_radians = attitude_data['pitch'] 
roll_radians = attitude_data['roll'] 

# 创建2×1布局的图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Robot Base Attitude Change', fontsize=16)

# 绘制俯仰角变化
ax1.plot(attitude_data['time'], pitch_radians, 'b-', linewidth=2)
ax1.set_ylabel('Pitch Angle (rad)', fontsize=12)
ax1.set_title('Pitch Angle Over Time', fontsize=14)
ax1.grid(True)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)  # 添加水平参考线

# 计算统计数据
pitch_mean = np.mean(pitch_radians)
pitch_std = np.std(pitch_radians)
pitch_max = np.max(np.abs(pitch_radians))
ax1.text(0.02, 0.95, f'Mean: {pitch_mean:.2f} rad\nStd Dev: {pitch_std:.2f} rad\nMax Offset: {pitch_max:.2f} rad', 
         transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7))

# 绘制滚转角变化
ax2.plot(attitude_data['time'], roll_radians, 'g-', linewidth=2)
ax2.set_xlabel('Time (sec)', fontsize=12)
ax2.set_ylabel('Roll Angle (rad)', fontsize=12)
ax2.set_title('Roll Angle Over Time', fontsize=14)
ax2.grid(True)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)  # 添加水平参考线

# 计算统计数据
roll_mean = np.mean(roll_radians)
roll_std = np.std(roll_radians)
roll_max = np.max(np.abs(roll_radians))
ax2.text(0.02, 0.95, f'Mean: {roll_mean:.2f} rad\nStd Dev: {roll_std:.2f} rad\nMax Offset: {roll_max:.2f} rad', 
         transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间

# 保存图表
output_file = os.path.splitext(os.path.basename(data_path))[0] + '_plot.png'
plt.savefig(output_file, dpi=300)
print(f"Saved visualization chart as: {output_file}")

# 显示图表
plt.show()

""" # 3D可视化
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot3D(attitude_data['time'], pitch_radians, roll_radians, 'gray')
ax.scatter3D(attitude_data['time'], pitch_radians, roll_radians, c=attitude_data['time'], cmap='viridis')
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Pitch Angle (rad)')
ax.set_zlabel('Roll Angle (rad)')
ax.set_title('3D Attitude Trajectory')

# 保存3D图表
output_file_3d = os.path.splitext(os.path.basename(data_path))[0] + '_3d_plot.png'
plt.savefig(output_file_3d, dpi=300)
print(f"Saved 3D visualization as: {output_file_3d}")

plt.show() """


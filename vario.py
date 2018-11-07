# coding:utf-8
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from scipy import optimize as opt
import matplotlib.pyplot as plt

# 標元データのファイル名を指定する
file_name = 'source.csv'
# 経験バリオグラムの階級幅を設定する
lag_h = 6
# フィッティングモデルを設定する
selected_model = 3
# フィッティングレンジを設定する
fitting_range = 60

# 元データを読み込み、配列に格納する
source_arr = np.genfromtxt(file_name, delimiter=',', skip_header=1)

# セミバリオグラムを計算する
def variogram(xyv_array):
	xy_dist = squareform(pdist(xyv_array[:, 0:2], 'euclidean'))
	s_vario = squareform(pdist(xyv_array[:, 2:3], 'euclidean')**2 / 2)
	return [xy_dist, s_vario]
z_vario = variogram(source_arr)

# 経験バリオグラム(階級幅の中でのセミバリオグラムの平均値)を計算する
def emp_variogram(z_vario, lag_h):
	num_rank = int(np.max(z_vario[0]) / lag_h)
	bin_means, bin_edges, bin_number = stats.binned_statistic(z_vario[0].flatten(), z_vario[1].flatten(), statistic='mean', bins=num_rank)
	e_vario = np.stack([bin_edges[1:], bin_means[0:]], axis=0)
	e_vario = np.delete(e_vario, np.where(e_vario[1] <= 0)[0], axis=1)
	return e_vario
e_vario = emp_variogram(z_vario, lag_h)

# 理論バリオグラムモデル
def liner_model(x, a, b):
	return a + b * x
def gaussian_model(x, a, b, c):
	return a + b * (1 - np.exp(-(x / c)**2))
def exponential_model(x, a, b, c):
	return a + b * (1 - np.exp(-(x / c)))
def spherical_model(x, a, b, c):
	cond = [x < c, x > c]
	func = [lambda x : a + (b / 2)  * (3 * (x / c) - (x / c)**3), lambda x : a + b]
	return np.piecewise(x, cond, func)

# モデルフィッティング
def auto_fit(e_vario, fitting_range, selected_model):
	# フィッティングレンジまでで標本バリオグラムを削る
	data = np.delete(e_vario, np.where(e_vario[0]>fitting_range)[0], axis=1)
	if (selected_model == 0):
		param, cov = opt.curve_fit(liner_model, data[0], data[1])
	elif (selected_model == 1):
		param, cov = opt.curve_fit(gaussian_model, data[0], data[1], [0, 0, fitting_range])
	elif (selected_model == 2):
		param, cov = opt.curve_fit(exponential_model, data[0], data[1], [0, 0, fitting_range])
	elif (selected_model == 3):
		param, cov = opt.curve_fit(spherical_model, data[0], data[1], [0, 0, fitting_range])
	return param
param = auto_fit(e_vario, fitting_range, selected_model)
param = np.insert(param, 0, fitting_range)
param = np.insert(param, 0, selected_model)
#np.savetxt("param.csv", param, fmt="%.10f", delimiter=',')

# グラフ作成
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(e_vario[0], e_vario[1], 'o')
xlim_arr = np.arange(0, np.max(e_vario[0]), lag_h)
if (param[0] == 0):
	ax.plot(xlim_arr, liner_model(xlim_arr, param[2], param[3]), 'r-')
	print(param[2], param[3])
elif (param[0] == 1):
	ax.plot(xlim_arr, gaussian_model(xlim_arr, param[2], param[3], param[4]), 'r-')
	print(xlim_arr, param[3], param[4])
elif (param[0] == 2):
	ax.plot(xlim_arr, exponential_model(xlim_arr, param[2], param[3], param[4]), 'r-')
	print(param[2], param[3], param[4])
elif (param[0] == 3):
	ax.plot(xlim_arr, spherical_model(xlim_arr, param[2], param[3], param[4]), 'r-')
	print(param[2], param[3], param[4])
# グラフのタイトルの設定
ax.set_title('Semivariogram')
# 軸ラベルの設定
ax.set_xlim([0, np.max(e_vario[0])])
ax.set_ylim([0, np.max(e_vario[1])])
ax.set_xlabel('Distance [m]')
ax.set_ylabel('Semivariance')
# グラフの縦横比を調整
aspect = 0.8 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])                     
ax.set_aspect(aspect)
# グラフの描画
plt.pause(15)

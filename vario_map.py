import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 元データを読み込み、配列に格納する
s_array = np.genfromtxt('source.csv', delimiter=',', skip_header=1)

# 異方性セミバリオグラム
def aniso_variogram(xyv_array):
    # 複素平面を用いる
    xy_array = xyv_array[:, 0] + xyv_array[:, 1] * 1j
    v_array  = xyv_array[:, 2]
    xy_array = np.tile(xy_array, (len(xy_array), 1))
    v_array  = np.tile(v_array, (len(v_array), 1))
    # 方位、距離、バリオグラムを求める  
    theta = np.angle(xy_array - xy_array.T)
    radii = abs(xy_array - xy_array.T)
    vario = abs(v_array - v_array.T)**2 / 2
    return theta, radii, vario

# バリオグラムマップ
def variogram_map(theta, radii, vario, grid_num):
    # 座標変換
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    z = vario.reshape(-1)
    # グリッドで分割し、補間する
    xgrid = ygrid = np.linspace(-np.amax(radii), np.amax(radii), grid_num)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    th = np.arctan2(ygrid, xgrid)
    ra = np.sqrt(xgrid**2 + ygrid**2)
    va = griddata((x.reshape(-1), y.reshape(-1)), z, (xgrid, ygrid), method='linear', fill_value=0)
    return th, ra, va

theta, radii, vario = aniso_variogram(s_array)
gr_th, gr_ra, gr_va = variogram_map(theta, radii, vario, 100)

# グラフを作成
fig = plt.figure()
ax  = plt.subplot(111, projection='polar')
plt.subplots_adjust(top=0.82, bottom=0.18)
ctf = ax.pcolormesh(gr_th, gr_ra, gr_va, cmap='jet')
# タイトルを設定
plt.text(0.5, 1.15, 'Semivariogram Map', horizontalalignment='center', transform=ax.transAxes, fontsize=16)
# グリッドを設定
ax.grid(True)
ax.set_rmax(np.amax(radii))
# カラーバーを設定
plt.colorbar(ctf, orientation='vertical', cax=plt.axes([0.85, 0.18, 0.02, 0.64]))
# Sliderを設定
c_slider = Slider(plt.axes([0.3, 0.01, 0.45, 0.03]), 'Color-Max', 0, np.amax(gr_va), valinit=np.amax(gr_va))
r_slider = Slider(plt.axes([0.3, 0.06, 0.45, 0.03]), 'Radius-Max', 0, np.amax(radii), valinit=np.amax(radii))
def slider_update(val):
    ctf.set_clim(0, c_slider.val)
    ax.set_rmax(r_slider.val)
    fig.canvas.draw_idle()
# Slider値変更時の処理の呼び出し
c_slider.on_changed(slider_update)
r_slider.on_changed(slider_update)
# グラフを表示
plt.show()
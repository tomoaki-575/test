# 必要なモジュールをインポート
import numpy as np
from scipy.special import factorial


# 式1 線形カメラモデル
def x_star(g, alpha, u_star):
# gはアナログ増幅由来のノイズ
# alphaは量子効率係数
# u_starはピクセル領域に検出される光子数の予測値
    return g * alpha * u_star


#　式２　ノイズ考慮した線形カメラモデル
def x(g, alpha, u, nd, nr):
# gはアナログ増幅由来のノイズ
# alphaは量子効率係数
# uはピクセル領域に検出される実際の光子数
# ndはダークノイズ
# nrは量子化のノイズ
    return g * (alpha * u + nd) + nr


# 式３　Uはu_starポアソン分布に従う
def U(lam, u_star):
# u_starはピクセル領域に検出される光子数の予測値
# lamはパラメータλ
    u_star = int(u_star)
    return np.power(lam, u_star) / factorial(u_star) * np.exp(-lam)


# Ndは期待値０、分散（σｄ）^2の正規分布に従う
def Nd(nd, sigma_d, mu_d=0):
    return 1 / np.sqrt(2 * np.pi * sigma_d) * np.exp(-(nd - mu_d)**2 / (2 * sigma_d**2))

# Nrは期待値0、分散(σr)^2の正規分布に従う
def Nr(nr, sigma_r, mu_r=0):
    return 1 / np.sqrt(2 * np.pi * sigma_r) * np.exp(-(nr-mu_r)**2 / (2 * sigma_r**2))

# Nは期待値mu、分散sigma^2の正規分布に従う
def N(n, mu, sigma_sqared):
    return 1 / np.sqrt(2 * np.pi * np.sqrt(sigma_sqared)) * np.exp(-(n - mu)**2 / (2 * sigma_sqared))


# 式4, 5　(式1, 2, 3)を組合せる
def x_transformed(g, alpha, u_star, lam, nd, nr, sigma_d, sigma_r):
# gはアナログ増幅由来のノイズ
# alphaは量子効率係数
# u_starはピクセル領域に検出される光子数の予測値
# lamはパラメータλ
# ndはダークノイズ
# nrは量子化のノイズ
# sigma_dはndが従う標準正規分布の標準偏差
# sigma_rはnrが従う標準正規分布の標準偏差
    k = g * alpha
    n = g * nd + nr
    sigma_sqared = g**2 * sigma_d**2 + sigma_r**2
    u_star = x_star(g, alpha, u_star) / k
    
    return k * U(lam, u_star) + N(n, 0, sigma_sqared)
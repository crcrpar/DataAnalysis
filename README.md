# サマーインターン（データサイエンス）選考課題

## 実装した手法
[J.Kim and H.Parkの"Fast nonnegative matrix factorization"](http://epubs.siam.org/doi/abs/10.1137/110821172)

### 特徴
協調フィルタリングによるレコメンドシステムにおいて，特徴量の数を$$k$$として，
$$\mathbb{R}^{m \times n}$$の行列を$$\mathbb{R}^{m \times k}と$$\mathbb{R}^{k \times n}$$
の積に分解する行列分解がメジャーである．
特に$$\mathbb{R}^{m \times n}$$の行列の要素は非負であることが多く，NMF（非負値行列分解）
という手法が適用できる．
NMFにはALSなど様々な手法があるが，ALSは計算量が非常に多い．そこで，今回は
上述した論文で紹介されている
ALSを元にしたBlock Principal Pivotingという手法を実装した．

### 参考文献
- 『集合知プログラミング』 Toby Segaran著，978-0-596-52932-1
- [NumPy Reference](http://docs.scipy.org/doc/numpy/reference/)
- [Python 2.7.x ドキュメント](http://docs.python.jp/2/index.html)

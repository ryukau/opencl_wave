# PyOpenCL初めの一歩
PyOpenCLでGPUを使って波のシミュレーションを行います。

# インストール
Debian busterにPyOpenCLをインストールします。

```bash
sudo apt install python3-pyopencl
```

nVidiaのGPUを使っている場合は次のパッケージをインストールします。

```bash
sudo apt install nvidia-opencl-icd nvidia-opencl-common nvidia-opencl-dev
```

<!--
AMDのGPUを使っている場合は次のパッケージをインストールします。

```bash
sudo apt install amd-opencl-icd amd-libopencl1 amd-opencl-dev
```
-->

`clinfo` をインストールします。 `clinfo` はOpenCLに対応しているデバイスの情報を表示します。

```bash
sudo apt clinfo
```

利用できるOpenCLのバージョンを確認します。

```bash
$ clinfo | grep "Device Version"
  Device Version                                  OpenCL 1.2 CUDA
  Device Version                                  OpenCL 1.2 pocl HSTR: pthread-x86_64-pc-linux-gnu-znver1
```

以降は OpenCL 1.2 を使います。

# OpenCL
OpenCL (Open Computing Language) はKhronos Groupによって定められた並列計算を行うAPIの仕様です。波のシミュレーションなどの計算はGPUで並列計算する方が速いことがあります。

OpenCLの実行速度はnVidiaのCUDAと比べると遅いそうですが、CPUなどでも動かせるという対応デバイスの多さが利点です。

## 仕様書とリファレンスカード
次のページからOpenCLの仕様書とリファレンスカードがダウンロードできます。

- [Khronos OpenCL Registry - The Khronos Group Inc](https://www.khronos.org/registry/OpenCL/)

仕様書の 2. GLOSSARY と 3. THE OPENCL ARCHITECTURE を読めば、大まかな仕組みがつかめます。OpenCLで使える機能を把握するには一覧になっているリファレンスカードが分かりやすいです。

## 大まかな仕組み
- 用語にリンク張る

OpenCLでやりたいことはカーネルの実行です。

カーネルはプログラムの中から呼び出される並列計算のプログラムで、見た目はC言語の関数です。カーネルを呼び出す側をホスト、カーネルを実行する側をデバイスと呼びます。ホストとデバイスをあわせて一括りにしたものをプラットフォームと呼びます。

使用するデバイスをまとめたものがコンテクストです。コンテクストにメモリを割り当ててカーネルを実行するコマンドを送ることで並列計算が行われます。

送られたコマンドはキューに格納されて順に実行されます。コマンドの種類にはカーネルの実行、メモリの操作（コピーや割り当て）、実行タイミングの同期があります。

メモリは配列として確保されます。配列に対してカーネルを実行する処理の単位としてワークアイテムとワークグループがあります。ワークアイテムは配列の一部にカーネルが1回適用される処理の単位です。ワークグループはいくつかのワークアイテムの集まりです。

すべてのワークグループとワークアイテムで共有されるメモリ領域をグローバルメモリ、一つのワークグループだけで共有されるメモリ領域をローカルメモリ、一つのワークアイテムの中だけで参照できるメモリ領域をプライベートメモリといいます。グローバルメモリの中でも、カーネルから値が変更できないように指定した領域のことをコンスタントメモリといいます。

OpenCLを使う手順は大まかに次のようになります。

1. プラットフォーム上で使用するデバイスをまとめてコンテクストを作る。
2. コンテクストで使うグローバルメモリを確保。
3. カーネルの書かれたプログラムをコンパイル。
4. コマンドキューにカーネルを実行するコマンドを送る。
5. コマンドキューにグローバルメモリをホストのメモリにコピーするコマンドを送る。

- [Understanding Kernels, Work-groups and Work-items — TI OpenCL Documentation](http://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html)
- [What is a host in opencl? - Stack Overflow](https://stackoverflow.com/questions/6485253/what-is-a-host-in-opencl)

# PyOpenCLの処理の流れ
必要なライブラリを `import` します。

```python
import numpy as np
import pyopencl as cl
```

Contextを作ります。Contextを作るにはPlatformからDeviceを取得して `pyopencl.Context` に渡します。

```python
platforms = cl.get_platforms()
devices = platforms[0].get_devices()
ctx = cl.Context(devices)
```

ContextとkernelのコードからProgramを作ります。

```python
prg = cl.Program(ctx, """
__kernel void index(__global float *array) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int len_x = get_global_size(0);
    const int index = x + len_x * y;
    array[index] = index;
}
""").build()
```

ContextとNumPyのarrayからBufferを作ってメモリを確保します。

```python
mf = cl.mem_flags
array = np.empty((10, 10)).astype(np.int32)
buf_array = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=array)
```

ContextからCommand-queueを作ります。

```python
queue = cl.CommandQueue(ctx)
```

Programからkernelを実行するコマンドを送ります。

```python
prg.index(queue, array.shape, None, buf_array)
```

ホストのメモリに計算結果をコピーします。

```python
cl.enqueue_copy(queue, array, buf_array)
```

以下はここまでのコードをまとめたものです。

```python
import numpy as np
import pyopencl as cl

platforms = cl.get_platforms()
devices = platforms[0].get_devices()
ctx = cl.Context(devices)

prg = cl.Program(ctx, """
__kernel void index(__global int *array) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int len_x = get_global_size(0);
    const int index = x + len_x * y;
    array[index] = index;
}
""").build()

mf = cl.mem_flags
array = np.empty((10, 10)).astype(np.int32)
buf_array = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=array)

queue = cl.CommandQueue(ctx)

prg.index(queue, array.shape, None, buf_array)

cl.enqueue_copy(queue, array, buf_array)
```

## 手軽にContextを取得
PyOpenCLの `create_some_context` を使えば適当にContextを作ることができます。 `create_some_context` を使ってContextを作るとホストプログラムの実行時にプラットフォームを選択するダイアログが出てきます。

```
$ python3 some_opencl.py
Choose platform:
[0] <pyopencl.Platform 'NVIDIA CUDA' at 0x12345687>
[1] <pyopencl.Platform 'Portable Computing Language' at 0x000011112222>
Choice [0]:0
Set the environment variable PYOPENCL_CTX='0' to avoid being asked again.
```

ダイアログに表示されているように `PYOPENCL_CTX` という名前のシェルの環境変数を設定することでダイアログを飛ばせます。bashの場合は以下のコマンドで設定できます。

```bash
echo "export PYOPENCL_CTX='0'" >> ~/.bashrc
source ~/.bashrc
```

## 2次元の波のシミュレーション
例として2次元の波をシミュレーションします。波動方程式から始めます。

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u
$$

計算したいのは $u$ の値です。FDM (Finite Difference Metdho) で離散化します。

$$
\frac{u_{t} - 2u_{t-1} + u_{t-2}}{\Delta_t^2}
= c^2 \frac{-2nu_{t} + \sum_{i = 0}^{n - 1} (u_{t,x_i - 1} + u_{t,x_i + 1})}{\Delta_x^2}
$$

- $n$ : 次元
- $t$ : 時間
- $x_i$ : 座標上の位置
- $c$ : 波の伝わる速度
- $\Delta_t$ : 1ステップごとに進める時間
- $\Delta_x$ : 格子の1辺の長さ

$u$ は $u(x_0, x_1, x_2, ..., x_n, t)$ の略です。数式では関数として表されますが、実装では $n$ 次元配列になります。参照している座標からの差分を $u_{x_0 + 1}$ のように下付き文字で表記しています。

$u_t$ の項を左辺、 $u_{t - n}$ の項を右辺に移行します。

$$
(1 + 2n c^2 \frac{\Delta_t^2}{\Delta_x^2}) u_{t}
- c^2 \frac{\Delta_t^2}{\Delta_x^2}
  \sum_{i = 0}^{n - 1} (u_{t,x_i - 1} + u_{t,x_i + 1})
= 2u_{t-1} - u_{t-2}
$$

定数をまとめて整理します。 $K, C_0, C_1$ は定数です。

$$
u_{t}
+ C_0 \sum_{i = 0}^{n - 1} (u_{t,x_i - 1} + u_{t,x_i + 1})
= (2u_{t-1} - u_{t-2}) / C_1
$$

$$
K = c^2 \frac{\Delta_t^2}{\Delta_x^2}
,\qquad
C_0 = - K / C_1
,\qquad
C_1 = 1 + 2n K
$$

実装します。Python3のホスト側のコードです。

```python
import imageio
import numpy as np
import pyopencl as cl
import pyopencl.cltypes


def read_cl(filename):
    with open(filename) as f:
        return f.read()


def normalize(array):
    array_min = np.min(array)
    return (array - array_min) / (np.max(array) - array_min)


def to_uint8(array):
    return (255.0 * normalize(array) + 1e-5).astype(np.uint8)


def write_mp4(result):
    imageio.mimwrite(
        "wave2d.mp4",
        to_uint8(np.array(result)),
        fps=60,
        codec="libx264",
        quality=9,
        # pixelformat="yuv420p",
    )


def iterate_wave2d(wave, boundary, C0, C1, attenuation, steps):
    result = []

    # OpenCL.
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, read_cl("wave2d.cl")).build()

    mem_flags = cl.mem_flags
    mem_flag_array = mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR

    buf_wave0 = cl.Buffer(ctx, mem_flag_array, hostbuf=wave[0])
    buf_wave1 = cl.Buffer(ctx, mem_flag_array, hostbuf=wave[1])
    buf_wave2 = cl.Buffer(ctx, mem_flag_array, hostbuf=wave[2])
    buf_boundary = cl.Buffer(ctx, mem_flag_array, hostbuf=boundary)
    buf_neighbor = cl.Buffer(ctx, mem_flags.READ_WRITE,
                             cl.cltypes.int4.itemsize * wave[0].size)

    prg.fill_neighbor_2d(queue, wave[0].shape, None, buf_neighbor)

    for step in range(steps):
        prg.wave2d_jacobi(
            queue,
            wave[0].shape,
            None,
            buf_boundary,
            buf_neighbor,
            buf_wave0,
            buf_wave1,
            buf_wave2,
            np.float32(C0),
            np.float32(C1),
            np.float32(attenuation),
            np.int32(512),
            np.int32(step),
        ).wait()

        cl.enqueue_copy(queue, wave[0], buf_wave0)
        result.append(np.copy(wave[0]))

    return result


def run_wave2d(
        shape=(256, 256),
        steps=1024,
        wave_speed=3,
        dx=0.1,
        dt=0.016666666666666666,
        attenuation=0.996):
    # Set constants.
    A = wave_speed * wave_speed * dt * dt / dx / dx

    C1 = 1 + 4 * A
    C0 = -A / C1

    # Initialize field.
    shape = np.array(shape)
    wave = np.zeros(np.hstack((3, shape))).astype(np.float32)

    # 0: field
    # 1: constant boundary or 0
    boundary = np.pad(
        np.zeros(shape - 2, np.uint8),
        [(1, 1) for _ in range(len(shape))],
        "constant",
        constant_values=1,
    )

    result = iterate_wave2d(wave, boundary, C0, C1, attenuation, steps)

    write_mp4(result)


if __name__ == "__main__":
    run_wave2d((256, 256), 1200, 4, 0.1, 1 / 60, 1)
```

OpenCLのカーネルのコードです。

```opencl
uint get_index_nd() {
    const int dimension = get_work_dim();
    int index = get_global_id(0);
    int size = 1;
    for (int dim = 1; dim < dimension; ++dim) {
        size *= get_global_size(dim - 1);
        index += size * get_global_id(dim);
    }
    return index;
}

__kernel void fill_boundary(__global uchar *boundary) {
    const int dimension = get_work_dim();
    for (int dim = 0; dim < dimension; ++dim) {
        int gid = get_global_id(dim);
        int size = get_global_size(dim);
        if (gid == 0 || gid == size - 1) {
            boundary[get_index_nd()] = 1;
            return;
        }
    }
}

__kernel void fill_neighbor_2d(__global uint4 *neighbor) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int len_x = get_global_size(0);
    const int index = x + len_x * y;

    neighbor[index].s0 = index - 1;
    neighbor[index].s1 = index + 1;
    neighbor[index].s2 = index - len_x;
    neighbor[index].s3 = index + len_x;
}

__kernel void wave2d_jacobi(
    __global uchar *boundary,
    __global uint4 *neighbor,
    __global float *wave0,
    __global float *wave1,
    __global float *wave2,
    const float C0,
    const float C1,
    const float attenuation,
    const int num_jacobi_iter,
    const int step
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int len_x = get_global_size(0);
    const int len_y = get_global_size(1);
    const int index = x + len_x * y;

    if (boundary[index] == 1) return;

    // Pick surface.
    if (step < 64 && x == len_x / 2 && y == len_y / 2)
        wave0[index] = 1.0;

    // Roll.
    wave2[index] = wave1[index];
    wave1[index] = attenuation * wave0[index];

    // Jacobi method.
    const float b = (2 * wave1[index] - wave2[index]) / C1;

    const int i_left = neighbor[index].s0;
    const int i_right = neighbor[index].s1;
    const int i_top = neighbor[index].s2;
    const int i_bottom = neighbor[index].s3;

    for (int i = 0; i < num_jacobi_iter; ++i) {
        wave0[index] = b - C0 * (
          wave0[i_left]
          + wave0[i_right]
          + wave0[i_top]
          + wave0[i_bottom]
        );
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

```

シミュレーションの中心は `wave2d_jacobi` カーネルです。境界 `boundary` と隣接するセルのインデックス `neighbor` を事前に計算しています。

数式の記号とコードの変数は次のように対応しています。

- $u_t$ - `wave0`
- $u_{t-1}$ - `wave1`
- $u_{t-2}$ - `wave2`
- $C_0$ - `C0`
- $C_1$ - `C1`

`wave2d_jacobi` の中身を見ていきます。

まずインデックスを取得します。

```opencl
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int len_x = get_global_size(0);
    const int len_y = get_global_size(1);
    const int index = x + len_x * y;
```

セルが波を伝える媒体かどうかを判定します。媒体でない部分は壁として扱っています。

```opencl
    if (boundary[index] == 1) return;
```

媒体をつついて波を起こします。

```opencl
    // Pick surface.
    if (step < 64 && x == len_x / 2 && y == len_y * step / 64)
        wave0[index] = 1.0;
```

時間ステップを一つ進めます。 `attenuation` は波が減衰するように適当に決めた定数です。

```opencl
    // Roll.
    wave2[index] = wave1[index];
    wave1[index] = attenuation * wave0[index];
```

ヤコビ法で方程式を解きます。 `b` は整理した式の右辺 $(2u_{t-1} - u_{t-2}) / C_1$ です。 `num_jacobi_iter` はヤコビ法の反復回数です。

```opencl
    // Jacobi method.
    const float b = (2 * wave1[index] - wave2[index]) / C1;

    const int i_left = neighbor[index].s0;
    const int i_right = neighbor[index].s1;
    const int i_top = neighbor[index].s2;
    const int i_bottom = neighbor[index].s3;

    for (int i = 0; i < num_jacobi_iter; ++i) {
        wave0[index] = b - C0 * (
          wave0[i_left]
          + wave0[i_right]
          + wave0[i_top]
          + wave0[i_bottom]
        );
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
```

<video controls>
  <source src="wave2d.mp4" type="video/mp4">
  <p>Video of 2d wave simulation.</p>
</video>

# 問題点
ここで紹介したヤコビ法の実装は遅いです。GPUで高速にSparse Matrixを解く方法としてConjugate GradientやMultigridが利用できるようです。

- [Sparse Matrix Solvers on the GPU: Conjugate Gradients and Multigrid - 28_GPUSim.pdf](http://www.cs.columbia.edu/cg/pdfs/28_GPUSim.pdf)

`shape=((512, 256))` など、正方形でない格子を使うと結果がおかしくなることがあります。

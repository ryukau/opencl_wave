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

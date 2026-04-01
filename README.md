# Fledge

Gravitational N-body test-particle simulation modeling massless dust particles
orbiting 1–3 massive bodies (single star, binary, or hierarchical triple).
Time integration uses leapfrog (kick-drift-kick). Binary orbits are solved
analytically via Kepler's equation; triple systems integrate the massive bodies
alongside the particles. Built on the [NEST](external/nest/) framework I/O layer
for checkpointing, timeseries, and snapshot output.

## Build

Requires C++23. On macOS, use Homebrew GCC (Apple Clang lacks OpenMP):

```bash
brew install gcc  # if not installed

cmake -B build -DNEST_BACKEND=CPU -DCMAKE_CXX_COMPILER=g++-14
cmake --build build
```

## Usage

```bash
./build/fledge                         # run with default config
./build/fledge tfinal=100 dt=0.0001    # override parameters
./build/fledge chkpt.0004.nest         # restart from checkpoint
./build/fledge chkpt.0004.nest tfinal=200  # restart with overrides
```

All config parameters can be set as `key=value` arguments on the command line.

### Examples

**Single star with a ring of particles:**

```bash
./build/fledge central_object_type=single num_particles=50 \
  inner_radius=1.0 outer_radius=1.0 tfinal=6.28 dt=0.001
```

**Equal-mass binary with a circumbinary disk:**

```bash
./build/fledge central_object_type=binary num_particles=100 \
  mass=1.0 q1=1.0 a1=0.5 inner_radius=2.0 outer_radius=4.0 \
  tfinal=10.0 dt=0.001
```

**Hierarchical triple (default config):**

```bash
./build/fledge central_object_type=triple num_particles=200 \
  tfinal=10.0 dt=0.0001
```

**Checkpoint round-trip:**

```bash
./build/fledge tfinal=5.0 dt=0.001 checkpoint_interval=1.0
./build/fledge chkpt.0005.nest tfinal=10.0
```

## Config Parameters

### Physics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tstart` | `0.0` | Initial simulation time |
| `tfinal` | `10.0` | Final simulation time |
| `dt` | `0.001` | Timestep |
| `softening` | `0.01` | Gravitational softening length |

### Central Object

| Parameter | Default | Description |
|-----------|---------|-------------|
| `central_object_type` | `"single"` | `"single"`, `"binary"`, or `"triple"` |
| `mass` | `1.0` | Total system mass |
| `q1` | `1.0` | Mass ratio of inner binary (binary/triple) |
| `a1` | `0.5` | Semi-major axis of inner binary (binary/triple) |
| `e1x` | `0.0` | Eccentricity vector x-component (binary/triple) |
| `e1y` | `0.0` | Eccentricity vector y-component (binary/triple) |
| `q2` | `0.001` | Mass ratio of outer orbit (triple only) |
| `a2` | `10.0` | Semi-major axis of outer orbit (triple only) |
| `e2x` | `0.0` | Outer eccentricity x-component (triple only) |
| `e2y` | `0.0` | Outer eccentricity y-component (triple only) |
| `inclination` | `0.0` | Orbital inclination in radians |

### Initial Conditions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_particles` | `1000` | Number of test particles |
| `setup_type` | `"random_disk"` | `"ring"`, `"random_disk"`, or `"uniform_disk"` |
| `ring_radius` | `1.5` | Radius for ring setup |
| `inner_radius` | `1.0` | Inner disk radius (random/uniform disk) |
| `outer_radius` | `2.0` | Outer disk radius (random/uniform disk) |
| `disk_center` | `"arbitrary"` | `"primary"`, `"secondary"`, or `"arbitrary"` (binary only) |
| `disk_center_x` | `0.0` | Arbitrary disk center x-coordinate |
| `disk_center_y` | `0.0` | Arbitrary disk center y-coordinate |
| `disk_center_z` | `0.0` | Arbitrary disk center z-coordinate |

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_interval` | `1.0` | Simulation time between checkpoints |
| `output_dir` | `"."` | Directory for output files |

## Output Files

| File | Contents |
|------|----------|
| `chkpt.NNNN.nest` | Full simulation state (positions, velocities, config) |
| `snap.NNNN.nest` | Particle position snapshot (x, y, z arrays) |
| `snap.NNNN.xdmf` | XDMF metadata for visualization |
| `timeseries.nest` | Time, particle count, and max radius per output step |

## Checkpoint Restart

To restart from a checkpoint, pass the `.nest` file as the first argument.
The config is restored from the checkpoint; additional `key=value` arguments
override restored values (e.g., extending `tfinal`).

```bash
./build/fledge chkpt.0005.nest tfinal=20.0
```

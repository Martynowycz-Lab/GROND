"""Write tiny.refl and tiny.expt, a synthetic DIALS reflection table and experiment list, with DIALS itself.

Run with the DIALS Python: <DIALS>/bin/python make_fixture.py OUTPUT_DIRECTORY. expected.json holds what
GROND 0.4.1 read from these files inside DIALS 3.30.0 (dxtbx and dials.array_family).
"""

import math
import sys
from pathlib import Path

from dials.algorithms.profile_model.gaussian_rs import Model
from dials.array_family import flex
from dxtbx.model import BeamFactory, Crystal, Detector, Experiment, ExperimentList, GoniometerFactory, ScanFactory
from scitbx import matrix

out = Path(sys.argv[1])
beam = BeamFactory.simple(0.019687)
detector = Detector()
root = detector.hierarchy()
root.set_frame((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (-2.5, 3.25, 0.0))
panel = detector.add_panel()
root.add_panel(panel)
panel.set_name("tiny")
panel.set_image_size((512, 512))
panel.set_pixel_size((0.075, 0.075))
panel.set_local_frame((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-19.2, -19.2, 960.0))
goniometer = GoniometerFactory.make_goniometer((1.0, 0.0, 0.0), (1, 0, 0, 0, 1, 0, 0, 0, 1))
goniometer.set_setting_rotation(matrix.col((0, 0, 1)).axis_and_angle_as_r3_rotation_matrix(2.0, deg=True).elems)
scan = ScanFactory.make_scan((1, 10), 0.5, (-30.0, 0.3), list(range(10)), deg=True)
rotation = matrix.col((0.3, 0.5, 0.8)).normalize().axis_and_angle_as_r3_rotation_matrix(37.0, deg=True)
a, c = 25.1, 18.3
cell = [rotation * matrix.col(v) for v in ((a, 0, 0), (-a / 2, a * math.sqrt(3) / 2, 0), (0, 0, c))]
crystal = Crystal(cell[0], cell[1], cell[2], space_group_symbol="R -3 c :H")
A = matrix.sqr(crystal.get_A())
crystal.set_A_at_scan_points([(matrix.col((1, 0, 0)).axis_and_angle_as_r3_rotation_matrix(0.3 * i, deg=True) * A).elems for i in range(11)])
profile = Model(None, 3, 0.0123, 0.234, deg=True)
experiments = ExperimentList([Experiment(beam=beam, detector=detector, goniometer=goniometer, scan=scan,
                                         crystal=crystal, profile=profile, identifier="tiny")])
experiments.as_file(str(out / "tiny.expt"))

n = 12
table = flex.reflection_table()
table["miller_index"] = flex.miller_index([(h, h - 3, 2 * h + 1) for h in range(-6, 6)])
table["id"] = flex.int([0] * n)
table["panel"] = flex.size_t([0] * n)
table["intensity.sum.value"] = flex.double([10.0 * i - 7.5 for i in range(n)])
table["intensity.sum.variance"] = flex.double([4.0 + i for i in range(n)])
table["intensity.prf.value"] = flex.double([9.0 * i for i in range(n)])
table["intensity.prf.variance"] = flex.double([3.0 + i for i in range(n)])
table["lp"] = flex.double([0.5 + 0.01 * i for i in range(n)])
table["dqe"] = flex.double([0.9] * n)
table["partiality"] = flex.double([1.0 - 0.05 * i for i in range(n)])
table["zeta"] = flex.double([0.1 * (i - 6) for i in range(n)])
table["d"] = flex.double([2.0 - 0.1 * i for i in range(n)])
table["bbox"] = flex.int6([(10 + i, 15 + i, 20, 26, i % 9, i % 9 + 2) for i in range(n)])
table["xyzcal.px"] = flex.vec3_double([(12.5 + i, 23.0, 0.5 + i % 9) for i in range(n)])
table["xyzobs.px.value"] = flex.vec3_double([(12.4 + i, 23.1, 0.6 + i % 9) if i != 3 else (math.nan,) * 3 for i in range(n)])
table["s1"] = flex.vec3_double([(0.0, 0.1, 50.0)] * n)
table["entering"] = flex.bool([i % 2 == 0 for i in range(n)])
table["shoebox"] = flex.shoebox(table["panel"], table["bbox"], allocate=True)
flags = table.flags
pattern = [flags.integrated_sum, flags.integrated, flags.integrated_sum | flags.overloaded, flags.dont_integrate,
           flags.integrated_sum | flags.foreground_includes_bad_pixels, flags.integrated_sum | flags.includes_bad_pixels,
           flags.integrated_sum | flags.in_powder_ring, flags.integrated_sum | flags.scaled,
           flags.integrated_sum | flags.outlier_in_scaling | flags.excluded_for_scaling, flags.integrated_sum | flags.bad_for_scaling,
           flags.integrated_sum | flags.failed_during_background_modelling, flags.integrated_sum | flags.failed_during_summation]
table["flags"] = flex.size_t([int(p) for p in pattern])
table.experiment_identifiers()[0] = "tiny"
table.as_file(str(out / "tiny.refl"))

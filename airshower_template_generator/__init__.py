from . import examples
from . import bins
from . import query
from . import plot
from . import model
from . import quality
from . import input_output
from . import projection
from . import production

import os
import glob
import json_numpy
import atmospheric_cherenkov_response


def init(
    work_dir,
    sites=None,
    particles=None,
    binning=examples.BINNING,
    run_config=examples.RUN_CONFIG,
):
    """
    Init a look-up-table.
    Create a working-directory for a new look-up-table.
    The working-dir stores the config, the map-and-reduce,
    and the final look-up-table.

    Parameters
    ----------
    work_dir : str, path
            The new working-directory.
    sites : dict
            The site(s) and its properties e.g. (CORSIKA-atmosphere-id,
            mag. field., altitude).
    particles : dict
            The particle(s) and its properties (CORSIKA-particle-id).
    binning : dict
            The binning of the look-up-table.
    run_config : dict
            Controlling the map-and-reduce, limiting the number of thrown
            shower.
    """

    if sites == None:
        sites = {}
        for sk in ["namibia"]:
            sites[sk] = atmospheric_cherenkov_response.sites.init(sk)

    if particles == None:
        particles = {}
        for pk in ["gamma"]:
            particles[sk] = atmospheric_cherenkov_response.particles.init(pk)

    os.makedirs(work_dir, exist_ok=True)
    json_numpy.write(path=os.path.join(work_dir, "sites.json"), out_dict=sites)
    json_numpy.write(
        path=os.path.join(work_dir, "particles.json"), out_dict=particles
    )
    json_numpy.write(
        path=os.path.join(work_dir, "binning.json"), out_dict=binning
    )
    json_numpy.write(
        path=os.path.join(work_dir, "run_config.json"), out_dict=run_config
    )


def populate(work_dir, multiprocessing_pool):
    """
    Populate the look-up-table
    """
    jobs = production.make_jobs(work_dir=work_dir)
    multiprocessing_pool.map(production.run_job, jobs)
    production.reduce(work_dir=work_dir)


def read(work_dir):
    """
    Read in the look-up-table.
    """
    paths = glob.glob(os.path.join(work_dir, "reduce", "*", "*", "raw.tar"))
    assert len(paths) == 1
    return input_output.read_raw(paths[0])

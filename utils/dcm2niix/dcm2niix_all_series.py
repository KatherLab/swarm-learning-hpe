#!/usr/bin/env python3

__author__ = 'Jeff'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = ['Marko van Treeck', 'Omar El Nahhas']
__email__ = 'markovantreeck@gmail.com'
__status__ = 'Development'

import logging
import multiprocessing

from tqdm import tqdm

'''
Common command line usage for our case -ba y -i y -f %i%t%d_%s -w y -o
-f is for naming the folder
%a=antenna (coil) name, %b=basename, %c=comments, %d=description, %e=echo number, %f=folder name, %i=ID of patient, %j=seriesInstanceUID, %k=studyInstanceUID, %m=manufacturer, %n=name of patient, %o=mediaObjectInstanceUID, %p=protocol, %r=instance number, %s=series number, %t=time, %u=acquisition number, %v=vendor, %x=study ID; %z=sequence name;
'
'''

import argparse
from pathlib import Path
import subprocess


def main(cohort_path: Path, outdir: Path) -> None:
    """Extracts tiles from whole slide images.
    Args:
        cohort_path:  A folder containing radiology data.
        outpath:  The output folder.
    """
    global series
    series_path_list = []
    outdir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=outdir / 'logfile', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    for patient_dir in (get_subdirs(cohort_path)):
        for series in (get_subdirs(get_subdirs(patient_dir)[0])):
            series_path_list.append(series)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for seires in tqdm(series_path_list):
        dcm2niix(outdir, seires)
        pool.apply_async(dcm2niix, (outdir, seires))
    '''
    pool = threadpool.ThreadPool(multiprocessing.cpu_count())
    requests = threadpool.makeRequests(dcm2niix(outdir, series), series_path_list)
    [pool.putRequest(req) for req in requests]
    pool.wait()
    '''


def dcm2niix(outpath: Path, dcm_path: Path) -> None:
    subprocess.run(["./bin/dcm2niix", "-ba", "y", "-i", "y", "-f", "%i%t%d_%s", "-w", "y", "-o", outpath, dcm_path])


def get_subdirs(path: Path) -> list:
    return [x for x in path.iterdir() if x.is_dir()]


def get_folder_with_specifc_naming(name: str, path: Path) -> bool:
    # get folder with 'ax dyn pre' in the name
    if name in path.name:
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert from dcm to niix.')
    parser.add_argument('-i', '--cohort_path', type=Path, metavar='', required=False,
                        default='/mnt/sda1/swarm-learning/radiology-dataset/manifest-1654812109500/Duke-Breast-Cancer-MRI/',
                        help='Path to the input directory.')
    parser.add_argument('-o', '--outdir', type=Path, metavar='', required=False,
                        default='/mnt/sda1/swarm-learning/radiology-dataset/converted-niix-all-series/',
                        help='Path to the output directory.')
    args = parser.parse_args()
    main(**vars(args))

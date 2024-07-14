#!/usr/bin/python3
import click
import logging
import sys
import os
import warnings
from timeit import default_timer as timer
import pkg_resources
from rich.logging import RichHandler
__version__ = pkg_resources.require("cognate")[0].version


# cli entry point
@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--verbosity', '-v', type=click.Choice(['info', 'debug']),
    default='info', help="Verbosity level, default = info.")
def cli(verbosity):
    """
    cognate - Copyright (C) 2024-2025 Hengchi Chen\n
    Contact: heche@psb.vib-ugent.be
    """
    logging.basicConfig(
        format='%(message)s',
        handlers=[RichHandler()],
        datefmt='%Y-%m-%d %H:%M:%S',
        level=verbosity.upper())
    logging.info("cognate v{}".format(__version__))
    pass


@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('data', type=click.Path(exists=True))
@click.option('--config_gff3', '-cg', default=None, show_default=True, help='configure file of gff3 if available')
@click.option('--tmpdir', '-tm', default=None, show_default=True, help='temporary working directory')
@click.option('--outdir', '-o', default='find_ortho', show_default=True, help='output directory')
@click.option('--to_stop', is_flag=True, help="don't translate through STOP codons")
@click.option('--cds', is_flag=True, help="enforce proper CDS sequences")
@click.option('--prot', is_flag=True, help="provided pep instead of cds sequences")
@click.option('--onlybhs', is_flag=True, help="only infer rbhs and bhs")
@click.option('--evalue', '-e', default=1e-10, help="e-value cut-off for similarity")
@click.option('--nthreads', '-n', default=4, show_default=True,help="number of threads to use")
@click.option('--iadhore_options', '-io', default=None, show_default=True,help="parameters in i-adhore")
@click.option('--pfam_dbhmm', default=None, show_default=True,help='profile for pfam hmm profile')
def find(**kwargs):
    """
    Find orthologues
    """
    _find(**kwargs)

def _find(data,config_gff3,tmpdir,outdir,to_stop,cds,prot,onlybhs,evalue,nthreads,iadhore_options,pfam_dbhmm):
    from cognate.ortho import cdsortho,syn_net,precluster_rbhfilter,mcl_cluster,rmalltmp
    start,syn = timer(),False
    gsmap,dmd_pairwise_outfiles,pep_paths,BHs,RBHs,tmpdir=cdsortho(data,tmpdir,outdir,to_stop,cds,evalue,nthreads,prot)
    if onlybhs:
        return
    if not (config_gff3 is None):
        syn_net(nthreads,dmd_pairwise_outfiles,iadhore_options,outdir,config_gff3)
        syn = True
    concatf = precluster_rbhfilter(RBHs,dmd_pairwise_outfiles,outdir,nthreads,evalue,pep_paths)
    mcl_cluster(dmd_pairwise_outfiles,outdir,syn,concatf=concatf)
    rmalltmp(tmpdir)
    end = timer()
    logging.info("Total run time: {} min".format(round((end-start)/60,2)))

if __name__ == '__main__':
    cli()

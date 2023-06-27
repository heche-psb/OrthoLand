import os
import uuid
import logging
from Bio import SeqIO
import subprocess as sp

def _mkdir(dirname):
    if not os.path.isdir(dirname) :
        os.mkdir(dirname)
    return dirname

def listdir(data):
    parent = os.getcwd()
    os.chdir(data)
    y = lambda x:os.path.join(data,x)
    files_clean = [y(i) for i in os.listdir() if i!="__pycache__"]
    os.chdir(parent)
    return files_clean

def writepepfile(fn,tmpdir,to_stop,cds):
    fname = os.path.join(tmpdir,os.path.basename(fn))
    with open(fname,'w') as f:
        for record in SeqIO.parse(fn, 'fasta'):
            aa_seq = record.translate(to_stop=to_stop, cds=cds, id=record.id)
            f.write(">{0}\n{1}\n".format(record.id,aa_seq.seq))
    return fname

def writepep(data,tmpdir,outdir,to_stop,cds):
    logging.info("Translating cds to pep")
    parent,pep_paths = os.getcwd(), []
    if tmpdir is None: tmpdir = "tmp_" + str(uuid.uuid4())
    _mkdir(tmpdir)
    fnames_seq = listdir(data)
    for fn in fnames_seq: pep_paths.append(writepepfile(fn,tmpdir,to_stop,cds))
    return pep_paths,tmpdir

def mkdb(pep_path,nthreads):
    cmd = ["diamond", "makedb", "--in", pep_path , "-d", pep_path, "-p", str(nthreads)]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    logging.debug(out.stderr.decode())

def pairdiamond(pep_path,pep_path_db,nthreads,evalue,outdir):
    dmd_folder = _mkdir(os.path.join(outdir,"diamond_results"))
    outfile = os.path.join(dmd_folder,"__".join([os.path.basename(pep_path),os.path.basename(pep_path)]) + ".tsv")
    cmd = ["diamond", "blastp", "-d", pep_path_db, "-q", pep_path, "-e", str(evalue), "-o", outfile, "-p", str(nthreads)]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    logging.debug(out.stderr.decode())

def pairwise_diamond(pep_paths,evalue,nthreads,outdir):
    pep_path_dbs = [i+'.dmnd' for i in pep_paths]
    logging.info("Running diamond")
    for pep_path in pep_paths: mkdb(pep_path,nthreads)
    for pep_path in pep_paths:
        for pep_path_db in pep_path_dbs:
            pairdiamond(pep_path,pep_path_db,nthreads,evalue,outdir)

def cdsortho(data,tmpdir,outdir,to_stop,cds,evalue,nthreads):
    """
    Infer orthogroups given cds sequences

    :param data: The directory containing cds sequences.
    :param tmpdir: The temporary working directory.
    :param outdir: The output directory.
    :param to_stop: Whether to translate through STOP codons, default False.
    :param cds: Whether to only translate the complete CDS, default False.
    :param evalue: The e-value cut-off for similarity, default 1e-10.
    :param nthreads: The number of threads to use, default 4.
    """
    _mkdir(outdir)
    pep_paths,tmppath = writepep(data,tmpdir,outdir,to_stop,cds)
    pairwise_diamond(pep_paths,evalue,nthreads,outdir)
    rmalltmp(tmppath)

def rmalltmp(tmppath):
    sp.run(['rm',tmppath,'-r'])

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
    gsmap = {}
    with open(fname,'w') as f:
        for record in SeqIO.parse(fn, 'fasta'):
            if not (gsmap.get(record.id) is None):
                logging.error("Duplicated gene id found in {}".format(os.path.basename(fn)))
                exit(1)
            gsmap[record.id] = os.path.basename(fn)
            aa_seq = record.translate(to_stop=to_stop, cds=cds, id=record.id)
            f.write(">{0}\n{1}\n".format(record.id,aa_seq.seq))
    return fname,gsmap

def writepep(data,tmpdir,outdir,to_stop,cds):
    logging.info("Translating cds to pep")
    parent,pep_paths,gsmaps = os.getcwd(),[],[]
    if tmpdir is None: tmpdir = "tmp_" + str(uuid.uuid4())
    _mkdir(tmpdir)
    fnames_seq = listdir(data)
    for fn in fnames_seq:
        pep_path,gsmap = writepepfile(fn,tmpdir,to_stop,cds)
        pep_paths.append(pep_path)
        before_ge = len(gsmaps)
        gsmaps.update(gsmap)
        after_ge = len(gsmaps)
        if after_ge-before_ge != len(gsmap):
            logging.error("Identical gene id found in {} with other sequence files".format(os.path.basename(fn)))
            exit(1)
    return pep_paths,tmpdir,gsmaps

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
    return outfile

def pairwise_diamond(pep_paths,evalue,nthreads,outdir):
    pep_path_dbs,outfiles = [i+'.dmnd' for i in pep_paths],{}
    logging.info("Running diamond")
    for pep_path in pep_paths: mkdb(pep_path,nthreads)
    for pep_path in pep_paths:
        for pep_path_db in pep_path_dbs:
            s1,s2=os.path.basename(pep_path),os.path.basename(pep_path_db)[:-5]
            outfiles["__".join(sorted([s1,s2]))]=pairdiamond(pep_path,pep_path_db,nthreads,evalue,outdir)
    return outfiles

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
    pep_paths,tmppath,gsmap = writepep(data,tmpdir,outdir,to_stop,cds)
    dmd_pairwise_outfiles = pairwise_diamond(pep_paths,evalue,nthreads,outdir,gsmaps)
    rmalltmp(tmppath)
    return gsmap,dmd_pairwise_outfiles

def syninfer(gsmap,dmd_pairwise_outfiles,parameters,gff3s,features,attributes,outdir):
    pairwise_iadhore(gsmap,dmd_pairwise_outfiles,parameters,gff3s,features,attributes,outdir)

def rmalltmp(tmppath):
    sp.run(['rm',tmppath,'-r'])

def getattr(s,attribute):
    for x in s.split(";"):
        y = x.split("=")
        if y[0].strip() == attribute:
            return y[1].strip()
    return ""

def gff2table(gff,feature,attribute,gsmap):
    """
    Read a GFF file to a pandas data frame, from a filename.
    """
    rows = []
    with open(gff, "r") as f:
        for l in f.readlines():
            if l.startswith("#") or l.strip("\n").strip()=='': continue
            x = l.strip("\n").strip("\t").split("\t")
            if x[2] == feature:
                a = getattr(x[-1], attribute)
                if a != "": rows.append({"gene": a, "scaffold": x[0], "start": int(x[3]), "or": x[6]})
    sp = gsmap[rows[0]["gene"]]
    df = pd.DataFrame.from_dict(rows).set_index("gene")
    return df,sp

def writegenelist(gff_info,dir_out):
    genelists = {}
    for scaffold,df_tmp in gff_info.groupby("scaffold"):
        fname = os.path.join(dir_out,scaffold)
        with open(fname, "w") as f:
            if len(list(df_tmp.index)) != len(set(df_tmp.index)):
                logging.error("There are duplicated gene IDs for given feature and attribute!")
                exit(1)
            for g in df_tmp.sort_values(by=["start"]).index: f.write(g+df_tmp.loc[g,"or"] + "\n")
        genelists[scaffold] = os.path.abspath(fname)
    return genelists

def writepair_iadhoreconf(sp_i,sp_j,gene_lists_i,gene_lists_j,parameters,dirname,pathbt):
    fname = os.path.join(dirname,"iadhore.conf")
    with open(fname,"w") as f:
        f.write("genome={}\n".format(sp_i))
        for scaf,path in gene_lists_i.items(): f.write(scaf+" "+path+"\n")
        f.write("\n")
        f.write("genome={}\n".format(sp_j))
        for scaf,path in gene_lists_j.items(): f.write(scaf+" "+path+"\n")
        f.write("\n")
        f.write("blast_table={}".format(pathbt))
        f.write("output_path={}".format(os.path.abspath(dirname)+"/iadhore-out"))
    para_dict = {"gap_size":30,"q_value":0.75,"cluster_gap":35,"prob_cutoff":0.01,"anchor_points":3,"alignment_method":"gg2","level_2_only":"false","multiple_hypothesis_correction":"FDR","visualizeGHM":"false","visualizeAlignment":"false"}
    if not (parameters is None):
        # "gap_size=30;q_value=0.75"
        for para in parameters.split(";"):
            key,value = para.split("=")
            if para_dict.get(key.strip()) is None:
                logging.error("The parameter {} is not included in i-adhore! Plesae double check".format(key.strip()))
            else: para_dict[key.strip()]=value.strip()
    with open(fname,"a") as f:
        for key,value in para_dict.items(): f.write(key,"=",str(value))
    return fname

def writeblastable(dmdtable,fname):
    df = pd.read_csv(dmdtable,header=None,index_col=None,sep='\t')
    for g1,g2 in zip(df[0],df[1]):
        with open(fname,"w") as f:
            f.write(g1+"\t"+g2+"\n")
    return os.path.abspath(fname)

def run_adhore(config_file):
    cmd = sp.run(['i-adhore', config_file], stderr=sp.PIPE, stdout=sp.PIPE)
    logging.warning(cmd.stderr.decode('utf-8'))
    logging.info(completed.stdout.decode('utf-8'))

def getgffinfo(config):
    gff3s,features,attributes=[],[],[]
    with open(config,"r") as f:
        for line in f.readlines():
            if len(line.split('\t')) !=3:
                logging.error("The format of config_gff3 file doesn't follow path\tfeature\tattribute. Please reformat!")
                exit(1)
            gff3,feature,attribute=line.split('\t')
            gff3s.append(gff3.strip("\n").strip())
            features.append(feature.strip("\n").strip())
            attributes.append(attribute.strip("\n").strip())
    return gff3s,features,attributes

def pairwise_iadhore(gsmap,dmd_pairwise_outfiles,parameters,gff3s,features,attributes,outdir,config=None):
    main_wd = _mkdir(os.path.join(outdir,"i-adhore"))
    gff_infos,gene_lists,sps = {},{},{} # sp follow the fname of sequence
    if not (config is None):
        gff3s,features,attributes=getgffinfo(config)
    for gff3,feature,attribute in zip(gff3s,features,attributes):
        splist_dir = _mkdir(os.path.join(main_wd,os.path.basename(gff3)+"_genelists"))
        gff_info,sp = gff2table(gff3,feature,attribute)
        sps[os.path.basename(gff3)] = sp
        gff_infos[os.path.basename(gff3)] = gff_info
        gene_lists[os.path.basename(gff3)] = writegenelist(gff_info,splist_dir)
    for i in range(len(gff3s)):
        for j in range(i,len(gff3s)):
            key_i,key_j = os.path.basename(gff3s[i]),os.path.basename(gff3s[j])
            sp_i,sp_j,gene_lists_i,gene_lists_j = sps[key_i],sps[key_j],gene_lists[key_i],gene_lists[key_j]
            dirname = _mkdir(os.path.join(main_wd,"__".join(sorted[sp_i,sp_j])))
            pathbt = writeblastable(dmd_pairwise_outfiles["__".join(sorted([sp_i,sp_j]))],os.path.join(dirname,"blast_table.txt"))
            fconf = writepair_iadhoreconf(sp_i,sp_j,gene_lists_i,gene_lists_j,parameters,dirname,pathbt)
            run_adhore(fconf)


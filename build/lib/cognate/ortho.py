import os
import uuid
import logging
from Bio import SeqIO
import subprocess as sp
import pandas as pd
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.optimize import basinhopping
import numpy as np
import sys
import itertools
from joblib import Parallel, delayed
from collections import ChainMap
from itertools import chain
from tqdm import tqdm,trange
import networkx as nx

sys.setrecursionlimit(10000000)

def _mkdir(dirname):
    if not os.path.isdir(dirname) :
        os.mkdir(dirname)
    return dirname

def infonoseed(postfix,form='whole'):
    logging.info("Searching in {}".format(form))
    if postfix==".FRBH":
        logging.info("No RBHs seed families were found!")
    else:
        logging.info("No syntenic seed families were found!")

def edgestonet(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G

def cmdhmmbuild(fpaln,fpalnhmm):
    cmd = ['hmmbuild'] + [fpalnhmm] + [fpaln]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def average_closeness_centrality(G):
    centrality = nx.closeness_centrality(G,distance='weight')
    return sum(centrality.values())/len(G)

def listdir(data):
    """
    Return the path to sequence files in a list
    """
    parent = os.getcwd()
    os.chdir(data)
    y = lambda x:os.path.join(data,x)
    files_clean = [y(i) for i in os.listdir() if i!="__pycache__"]
    logging.info("In total {} sequence files are found".format(len(files_clean)))
    logging.info(", ".join([os.path.basename(i) for i in files_clean]))
    os.chdir(parent)
    return files_clean

def writepepfile(fn,tmpdir,to_stop,cds,protein):
    """
    Return file names of translated cds or given pep, map file and gene length dict for normalization
    """
    # Here we require the species name contains no "__" which might cause error afterwards in "s1__s2"
    fname = os.path.join(tmpdir,os.path.basename(fn))
    gsmap,gldict = {},{}
    with open(fname,'w') as f:
        for record in tqdm(SeqIO.parse(fn, 'fasta'),desc="Reading {}".format(os.path.basename(fn)),unit=" sequences"):
            if not (gsmap.get(record.id) is None):
                logging.error("Duplicated gene id {0} found in {1}".format(record.id,os.path.basename(fn)))
                exit(1)
            gsmap[record.id] = os.path.basename(fn)
            aa_seq = record.translate(to_stop=to_stop, cds=cds, id=record.id) if not protein else record
            gldict[record.id] = len(aa_seq) if not protein else len(aa_seq.seq)
            f.write(">{0}\n{1}\n".format(record.id,aa_seq.seq))
    return fname,gsmap,gldict

def writepep(data,tmpdir,outdir,to_stop,cds,protein):
    if not protein: logging.info("Checking cds files")
    else: logging.info("Checking protein files")
    parent,pep_paths,gsmaps,gldicts = os.getcwd(),[],{},{}
    if tmpdir is None: tmpdir = "tmp_" + str(uuid.uuid4())
    _mkdir(tmpdir)
    fnames_seq = listdir(data)
    for fn in fnames_seq:
        pep_path,gsmap,gldict = writepepfile(fn,tmpdir,to_stop,cds,protein)
        pep_paths.append(pep_path)
        before_ge = len(gsmaps)
        gsmaps.update(gsmap)
        gldicts.update(gldict)
        after_ge = len(gsmaps)
        if after_ge-before_ge != len(gsmap):
            logging.error("Identical gene id found in {} with other sequence files".format(os.path.basename(fn)))
            exit(1)
    return pep_paths,tmpdir,gsmaps,gldicts

def mkdb(pep_path,nthreads):
    cmd = ["diamond", "makedb", "--in", pep_path , "-d", pep_path, "-p", str(nthreads)]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    logging.debug(out.stderr.decode())

def addgl(outfile,gldict):
    df = pd.read_csv(outfile,header=None,index_col=None,sep='\t')
    df[12] = [gldict[g1]*gldict[g2] for g1,g2 in zip(df[0],df[1])]
    return df

def addnormscore(df):
    # Here I used overall hits without subdividing bins
    slope, intercept, r, p, se = stats.linregress(np.log10(df[12]), np.log10(df[11]))
    df[13] = [j/(pow(10, intercept)*(l**slope)) for j,l in zip(df[11],df[12])]
    return df

def pairdiamond(pep_path,pep_path_db,nthreads,evalue,outdir,gldict):
    dmd_folder = _mkdir(os.path.join(outdir,"diamond_results"))
    outfile = os.path.join(dmd_folder,"__".join([os.path.basename(pep_path),os.path.basename(pep_path_db)[:-5]]) + ".tsv")
    logging.info("{0} vs. {1}".format(os.path.basename(pep_path),os.path.basename(pep_path_db)[:-5]))
    cmd = ["diamond", "blastp", "-d", pep_path_db, "-q", pep_path, "-e", str(evalue), "-o", outfile, "-p", str(nthreads)]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    df = addgl(outfile,gldict)
    df = addnormscore(df)
    norm_outfile = outfile[:-4]+".norm.tsv"
    df.to_csv(norm_outfile,header=False,index=False,sep='\t')
    logging.debug(out.stderr.decode())
    return norm_outfile

def multi_pairdiamond(pep_path_db,pep_path,nthreads,evalue,outdir,gldict):
    """
    Return a dict with key as s1__s2 and value as norm outfile
    """
    s1,s2=os.path.basename(pep_path),os.path.basename(pep_path_db)[:-5]
    d = {"__".join(sorted([s1,s2])):pairdiamond(pep_path,pep_path_db,nthreads,evalue,outdir,gldict)}
    return d

def pairwise_diamond(pep_paths,evalue,nthreads,outdir,gldict):
    pep_path_dbs,outfiles = [i+'.dmnd' for i in pep_paths],{}
    logging.info("Running diamond and normalization using {} threads".format(nthreads))
    for pep_path in pep_paths: mkdb(pep_path,nthreads)
    idenfiers = []
    for i,pep_path in enumerate(pep_paths):
        for j in range(i,len(pep_paths)): idenfiers.append((j,pep_path))
    ds = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(multi_pairdiamond)(pep_path_dbs[j],pep_path,nthreads,evalue,outdir,gldict) for j,pep_path in idenfiers)
    for d in ds: outfiles.update(d)
    return outfiles

def cdsortho(data,tmpdir,outdir,to_stop,cds,evalue,nthreads,prot):
    """
    Running diamond given cds or pep sequences

    :param data: The directory containing cds sequences.
    :param tmpdir: The temporary working directory.
    :param outdir: The output directory.
    :param to_stop: Whether to translate through STOP codons, default False.
    :param cds: Whether to only translate the complete CDS, default False.
    :param evalue: The e-value cut-off for similarity, default 1e-10.
    :param nthreads: The number of threads to use, default 4.
    :param prot: Input as protein data instead of cds, default False.
    """
    _mkdir(outdir)
    pep_paths,tmppath,gsmap,gldict = writepep(data,tmpdir,outdir,to_stop,cds,prot)
    dmd_pairwise_outfiles = pairwise_diamond(pep_paths,evalue,nthreads,outdir,gldict)
    logging.info("Diamond done")
    BHs = get_bhs(dmd_pairwise_outfiles,outdir,gsmap)
    logging.info("BHs retrieved")
    RBHs = get_rbhs(dmd_pairwise_outfiles,outdir,gsmap)
    logging.info("RBHs retrieved")
    #rmalltmp(tmppath)
    return gsmap,dmd_pairwise_outfiles,pep_paths,BHs,RBHs,tmppath

def get_bhs(dmd_pairwise_outfiles,outdir,gsmap):
    BHs = {}
    BH_path = _mkdir(os.path.join(outdir,"BHs"))
    y = lambda x:os.path.join(BH_path,"__".join(sorted([x[0],x[1]]))+".BH")
    for spair,fn in dmd_pairwise_outfiles.items():
        spair_list = spair.split("__")
        if spair_list[0] == spair_list[1]: continue
        Processor_DMD(fn,outpath=y(spair_list),gs_map=gsmap).write_bh()
        BHs[spair]=y(spair_list)
    return BHs

def get_rbhs(dmd_pairwise_outfiles,outdir,gsmap):
    RBHs = {}
    RBH_path = _mkdir(os.path.join(outdir,"RBHs"))
    y = lambda x:os.path.join(RBH_path,"__".join(sorted([x[0],x[1]]))+".RBH")
    for spair,fn in dmd_pairwise_outfiles.items():
        spair_list = spair.split("__")
        if spair_list[0] == spair_list[1]: continue
        Processor_DMD(fn,outpath=y(spair_list),gs_map=gsmap).write_rbh()
        RBHs[spair]=y(spair_list)
    return RBHs

def loop_until_nomore(Df,df,N_Lim,gene_list):
    Orig_num = Df.shape[0]
    gene_pools = list(set(Df[0].to_list() + Df[1].to_list()))
    #for gene in gene_pools: Df = pd.concat([df[df[0] == gene],df[df[1] == gene],Df]).drop_duplicates()
    for gene in sorted(gene_pools, key = lambda x:x in gene_list, reverse=True):
        if Df.shape[0] >= N_Lim:
            return Df
        Df = pd.concat([df[(df[0] == gene) | (df[1] == gene)],Df]).drop_duplicates()
    if Orig_num == Df.shape[0]:
        return Df
    else:
        return loop_until_nomore(Df,df,N_Lim,gene_list) 

def getalledges(gene1,gene2,df,ctf):
    Df = df[(df[0].isin([gene1, gene2])) | (df[1].isin([gene1, gene2]))]
    #Df = pd.concat([df[df[0] == gene1],df[df[1] == gene1],df[df[0] == gene2],df[df[1] == gene2]]).drop_duplicates()
    Df = loop_until_nomore(Df,df)
    # Here are all the edge pairs with score larger than ctf
    edges = [(g1,g2,score) for g1,g2,score in zip(Df[0],Df[1],Df[13]) if score >= ctf]
    return edges

def getalledges_whole(gene_list,df,ctf,N,lim=100):
    N_Lim = N*lim
    Df = df[(df[0].isin(gene_list)) | (df[1].isin(gene_list))]
    #Df = pd.concat([df[df[0] == gene1],df[df[1] == gene1],df[df[0] == gene2],df[df[1] == gene2]]).drop_duplicates()
    Df = loop_until_nomore(Df,df,N_Lim,gene_list)
    # Here are all the edge pairs with score larger than ctf
    all_edges = [(g1,g2,score) for g1,g2,score in zip(Df[0],Df[1],Df[13])]
    edges = [(g1,g2,score) for g1,g2,score in all_edges if score >= ctf]
    if len(edges) == 0:
        edges = [sorted(all_edges, key=lambda x:x[2])[-1]]
    return edges

class Processor_DMD:
    """
    Processor of diamond result
    """
    def __init__(self,fname,outpath=None,gs_map=None):
        self.fname = os.path.abspath(fname)
        self.outpath = outpath
        self.gsmap = gs_map

    def readdf(self):
        self.df = pd.read_csv(self.fname,header=None,index_col=None,sep='\t',usecols=[0,1,13])

    def write_rbh(self):
        self.readdf()
        df1 = self.df.sort_values(13,ascending=False).drop_duplicates([0])
        df2 = self.df.sort_values(13,ascending=False).drop_duplicates([1])
        self.rbh = df1.merge(df2)
        sp_0,sp_1 = self.gsmap[list(self.rbh[0])[0]],self.gsmap[list(self.rbh[1])[0]]
        self.rbh = self.rbh.rename(columns={0:sp_0,1:sp_1,13:"Score_{0}_{1}".format(sp_0,sp_1)})
        self.rbh.to_csv(self.outpath,header=True,index=False,sep='\t')

    def write_bh(self):
        self.readdf()
        df1 = self.df.sort_values(13,ascending=False).drop_duplicates([0])
        df2 = self.df.sort_values(13,ascending=False).drop_duplicates([1])
        self.bh = pd.concat([df1,df2]).drop_duplicates()
        sp_0,sp_1 = self.gsmap[list(self.bh[0])[0]],self.gsmap[list(self.bh[1])[0]]
        self.bh = self.bh.rename(columns={0:sp_0,1:sp_1,13:"Score_{0}_{1}".format(sp_0,sp_1)})
        self.bh.to_csv(self.outpath,header=True,index=False,sep='\t')

    def get_rbh_score(self):
        self.rbh = pd.read_csv(self.fname,header=0,index_col=None,sep='\t')
        self.ctf = {(gx,gy):score for gx,gy,score in zip(self.rbh.iloc[:,0],self.rbh.iloc[:,1],self.rbh.iloc[:,2])}
        self.pair = list(self.ctf.keys())
        return self.pair,self.ctf

class DMD_ABC:
    """
    Manipulater of diamond results in ABC format given seed orthologues
    """
    def __init__(self,fname,seedortho,nthreads,outpath,ctf=None):
        self.fname = os.path.abspath(fname)
        self.seedortho = seedortho
        self.nthreads = nthreads
        self.outpath = outpath
        self.ctf = ctf

    def readdf(self):
        self.df = pd.read_csv(self.fname,header=None,index_col=None,sep='\t',usecols=[0,1,13])

    def filterdf_getctf(self):
        self.readdf()
        self.df,self.ctf = filter_interfamedge_getctf(self.seedortho,self.df,self.nthreads)
        self.df.to_csv(self.outpath,header=False,index=False,sep='\t')
        return self.ctf

    def filterdf(self):
        self.readdf()
        self.df = filter_interfamedge(self.seedortho,self.df,self.ctf,self.nthreads)
        self.df.to_csv(self.outpath,header=False,index=False,sep='\t')

def rmalltmp(tmppath):
    sp.run(['rm',tmppath,'-r'])

def getattr(s,attribute):
    for x in s.split(";"):
        y = x.split("=")
        if y[0].strip() == attribute:
            return y[1].strip()
    return ""

def gff2table(gff,feature,attribute):
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
    df = pd.DataFrame.from_dict(rows).set_index("gene")
    return df

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

def writepair_iadhoreconf(sp_i,sp_j,gene_lists_i,gene_lists_j,parameters,dirname,pathbt,nthreads):
    fname = os.path.join(dirname,"iadhore.conf")
    para_dict = {"gap_size":30,"q_value":0.75,"cluster_gap":35,"prob_cutoff":0.01,"anchor_points":3,"alignment_method":"gg2","level_2_only":"false","multiple_hypothesis_correction":"FDR","visualizeGHM":"false","visualizeAlignment":"false","number_of_threads":nthreads}
    if not (parameters is None):
        for para in parameters.split(";"):
            key,value = para.split("=")
            if para_dict.get(key.strip()) is None:
                logging.error("The parameter {} is not included in i-adhore! Plesae double check".format(key.strip()))
            else: para_dict[key.strip()]=value.strip()
    with open(fname,"w") as f:
        f.write("genome={}\n".format(sp_i))
        for scaf,path in gene_lists_i.items(): f.write(scaf+" "+path+"\n")
        f.write("\n")
        f.write("genome={}\n".format(sp_j))
        for scaf,path in gene_lists_j.items(): f.write(scaf+" "+path+"\n")
        f.write("\n")
        for key,value in para_dict.items(): f.write(key+"="+str(value)+"\n")
        f.write("blast_table={}\n".format(pathbt))
        f.write("output_path={}\n".format(os.path.abspath(dirname)+"/iadhore-out"))
    return fname

def writeblastable(dmdtable,fname):
    df = pd.read_csv(dmdtable,header=None,index_col=None,sep='\t')
    with open(fname,"w") as f:
        for g1,g2 in zip(df[0],df[1]): f.write(g1+"\t"+g2+"\n")
    return os.path.abspath(fname)

def run_adhore(config_file):
    cmd = sp.run(['i-adhore', config_file], stderr=sp.PIPE, stdout=sp.PIPE)
    logging.debug(cmd.stderr.decode('utf-8'))
    logging.debug(cmd.stdout.decode('utf-8'))

def getgffinfo(config):
    gff3s,features,attributes,sps=[],[],[],[]
    with open(config,"r") as f:
        for line in f.readlines():
            if len(line.split('\t')) !=4:
                logging.error("The format of config_gff3 file doesn't follow seqname\tgffpath\tfeature\tattribute. Please reformat!")
                exit(1)
            sp,gff3,feature,attribute=line.split('\t')
            sps.append(sp.strip())
            gff3s.append(gff3.strip())
            features.append(feature.strip())
            attributes.append(attribute.strip("\n").strip())
    return sps,gff3s,features,attributes

def multi_run_adhore(i,j,gff3s,sps,gene_lists,main_wd,dmd_pairwise_outfiles,parameters,nthreads):
    key_i,key_j = os.path.basename(gff3s[i]),os.path.basename(gff3s[j])
    sp_i,sp_j,gene_lists_i,gene_lists_j = sps[key_i],sps[key_j],gene_lists[key_i],gene_lists[key_j]
    logging.info("{0} vs. {1}".format(sp_i,sp_j))
    dirname = _mkdir(os.path.join(main_wd,"__".join(sorted([sp_i,sp_j]))))
    pathbt = writeblastable(dmd_pairwise_outfiles["__".join(sorted([sp_i,sp_j]))],os.path.join(dirname,"blast_table.txt"))
    fconf = writepair_iadhoreconf(sp_i,sp_j,gene_lists_i,gene_lists_j,parameters,dirname,pathbt,nthreads)
    run_adhore(fconf)
    return {(sp_i,sp_j):os.path.join(dirname,"iadhore-out","multiplicon_pairs.txt")}

def mcl_cluster(dmd_pairwise_outfiles,outdir,syn,concatf=None):
    if concatf is not None:
        mcl(concatf).run_mcl()
        return
    ABC = mergeallF(dmd_pairwise_outfiles,outdir,syn)
    ABC.drop_duplicates()
    ABC.concat()
    mcl(ABC.ABC).run_mcl()

class mergeallF:
    """
    Merge all filtered hits
    """
    def __init__(self,dmd_pairwise_outfiles,outdir,syn):
        self.dmd_fs = dmd_pairwise_outfiles
        self.wd = _mkdir(os.path.join(outdir,"MCL"))
        self.syn = syn

    def readf(self,fn):
        df = pd.read_csv(fn,header=None,index_col=None,sep='\t')
        return df

    def drop_duplicates(self):
        self.dmd_fs_merge = {}
        y = lambda x:os.path.join(*x.split("/")[:-2],"i-adhore","syntelog",os.path.basename(x))
        if not self.syn:
            for key,value in self.dmd_fs.items(): self.dmd_fs_merge[key] = value + ".FRBH"
        else:
            for key,value in self.dmd_fs.items():
                if not os.path.exists(y(value)+".FSYN"):
                    self.dmd_fs_merge[key] = value + ".FRBH"
                else:
                    df_RBH = self.readf(value+".FRBH")
                    df_SYN = self.readf(y(value)+".FSYN")
                    df_RBH.merge(df_SYN).to_csv(value+".ABC",header=False,index=False,sep='\t')
                    self.dmd_fs_merge[key] = value+".ABC"

    def concat(self):
        cmd = ["cat"] + [fn for fn in self.dmd_fs_merge.values()]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        self.ABC = os.path.join(self.wd,"All_hits.ABC")
        with open(self.ABC, 'w') as f: f.write(out.stdout.decode('utf-8'))


class mcl:
    """
    MCL clustering
    """
    def __init__(self,graph_file):
        self.graph_file = graph_file

    def run_mcl(self, inflation=2):
        f1 = self.graph_file
        f2 = f1 + ".tab"
        f3 = f1 + ".mci"
        f4 = f2 + ".I{}".format(inflation*10)
        outfile = f1 + ".mcl"
        command = ['mcxload', '-abc', f1, '--stream-mirror', '-o', f3, '-write-tab', f2]
        sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)
        command = ['mcl', f3, '-I', str(inflation), '-o', f4]
        sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)
        command = ['mcxdump', '-icl', f4, '-tabr', f2, '-o', outfile]
        sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)
        return outfile

def syn_net(nthreads,dmd_pairwise_outfiles,parameters,outdir,gffconf):
    syn = SYN_Net(gffconf,dmd_pairwise_outfiles,parameters,outdir,nthreads)
    syn.write_gene_list()
    syn.run_iadhore()
    syn.connectsyn()
    logging.info("Connecting syntenic components")
    ConnectPair = connectpairs(syn.pairs_scores,syn.wd_sy,nthreads)
    ConnectPair.writedf(ConnectPair.mergedf(),"Syn.Net")
    logging.info("Discarding low-similarity edges")
    ConnectPair.filterhits_pairwise_heuristic(syn.dmd_fs,postfix=".FSYN")
    #ConnectPair.filterhits_pairwise(syn.dmd_fs,postfix=".FSYN")

class SYN_Net:
    """
    Infer the syntenic network
    """
    def __init__(self,gffconf,dmd_fs,parameters,outdir,nthreads):
        self.gffconf = gffconf
        self.dmd_fs = dmd_fs
        self.parameters = parameters
        self.nthreads = nthreads
        self.wd = _mkdir(os.path.join(outdir,"i-adhore"))
        self.wd_gl = _mkdir(os.path.join(self.wd,"genelist"))
        self.wd_sy = _mkdir(os.path.join(self.wd,"syntelog"))

    def write_gene_list(self):
        sps,self.gff3s,features,attributes=getgffinfo(self.gffconf)
        self.gene_lists,self.SPS = {},{}
        for sp,gff3,feature,attribute in zip(sps,self.gff3s,features,attributes):
            splist_dir = _mkdir(os.path.join(self.wd_gl,os.path.basename(gff3)))
            gff_info = gff2table(gff3,feature,attribute)
            self.SPS[os.path.basename(gff3)] = sp
            self.gene_lists[os.path.basename(gff3)] = writegenelist(gff_info,splist_dir)

    def run_iadhore(self):
        self.aps,identifers = {},[]
        logging.info("Running i-adhore using {} threads".format(self.nthreads))
        for i in range(len(self.gff3s)):
            for j in range(i+1,len(self.gff3s)):
                identifers.append((i,j))
        ds = Parallel(n_jobs=self.nthreads,backend='multiprocessing')(delayed(multi_run_adhore)(i,j,self.gff3s,self.SPS,self.gene_lists,self.wd,self.dmd_fs,self.parameters,self.nthreads) for i,j in identifers)
        for d in ds: self.aps.update(d)
        logging.info("I-adhore done")

    def connectsyn(self):
        y = lambda x:"__".join(sorted([x[0],x[1]]))
        pairs = {key:getapair(key,value) for key,value in self.aps.items()}
        self.pairs_scores = {y(key):score_merger(self.dmd_fs[y(key)],value,self.wd_sy) for key,value in pairs.items()}

def score_merger(dmd_fn,df_pair,wd):
    fn = os.path.join(wd,"__".join(sorted(df_pair.columns))+".Syn")
    if len(df_pair) == 0:
        df_pair["Score_{0}_{1}".format(df_pair.columns[0],df_pair.columns[1])] = []
        df_pair.to_csv(fn,header=True,index=False,sep='\t')
    else:
        df_score = pd.read_csv(dmd_fn,header=None,index_col=None,sep='\t',usecols=[0,1,13])
        if df_pair.iloc[0,0] in list(df_score[0]):
            df_score = df_score.rename(columns={0:df_pair.columns[0],1:df_pair.columns[1]})
        else:
            df_score = df_score.rename(columns={0:df_pair.columns[1],1:df_pair.columns[0]})
        df_score = df_score.rename(columns={13:"Score_{0}_{1}".format(df_score.columns[0],df_score.columns[1])})
        df_pair.merge(df_score).to_csv(fn,header=True,index=False,sep='\t')
    return fn

def getapair(spair,fn):
    df = pd.read_csv(fn,sep="\t",index_col=0)
    if len(df.columns) == 5: df = df.drop(columns=['gene_y']).rename(columns = {'gene_x':'gene_y'}).rename(columns = {'Unnamed: 2':'gene_x'})
    df['gene_xy'] = ["__".join(sorted([gx,gy])) for gx,gy in zip(df['gene_x'],df['gene_y'])]
    df = df.drop_duplicates(subset=['gene_xy'])
    df_pair = df.loc[:,['gene_x','gene_y']].rename(columns={'gene_x':spair[1],'gene_y':spair[0]})
    return df_pair

def syninfer(nthreads,gsmap,dmd_pairwise_outfiles,parameters,gff3s,features,attributes,outdir,config=None):
    main_wd = _mkdir(os.path.join(outdir,"i-adhore"))
    gff_infos,gene_lists,sps,aps,identifers = {},{},{},{},[] # sp follow the fname of sequence
    if not (config is None):
        gff3s,features,attributes=getgffinfo(config)
    logging.info("Writing genelists for i-adhore")
    for gff3,feature,attribute in zip(gff3s,features,attributes):
        splist_dir = _mkdir(os.path.join(main_wd,os.path.basename(gff3)+"_genelists"))
        gff_info,sp = gff2table(gff3,feature,attribute,gsmap)
        sps[os.path.basename(gff3)] = sp
        gff_infos[os.path.basename(gff3)] = gff_info
        gene_lists[os.path.basename(gff3)] = writegenelist(gff_info,splist_dir)
    logging.info("Running i-adhore using {} threads".format(nthreads))
    for i in range(len(gff3s)):
        for j in range(i+1,len(gff3s)):
            identifers.append((i,j))
    ds = Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11)(delayed(multi_run_adhore)(i,j,gff3s,sps,gene_lists,main_wd,dmd_pairwise_outfiles,parameters,nthreads) for i,j in identifers)
    for d in ds: aps.update(d)
    logging.info("I-adhore done")
    return aps

def checkpress(pfam_dbhmm):
    if not os.path.isfile(pfam_dbhmm+'.h3p'):
        cmd = ['hmmpress'] + [pfam_dbhmm]
        sp.run(cmd,stdout=sp.PIPE,stderr=sp.PIPE)

def pfamscan(pfam_dbhmm,pep_paths,nthreads,evalue,outdir):
    logging.info("Starting protein domain search")
    domain_dir = _mkdir(os.path.join(outdir,'pfam_domain'))
    cmds = []
    y = lambda x:os.path.join(domain_dir,x)
    checkpress(pfam_dbhmm)
    for pep_path in pep_paths:
        cmd = ['hmmscan', '--tblout', y(os.path.basename(pep_path)+".tbl"), '--pfamtblout', y(os.path.basename(pep_path)+'.pfam'), '--noali', '-E', '{}'.format(evalue), pfam_dbhmm, pep_path]
        cmds.append(cmd)
    Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11)(delayed(sp.run)(cmd,stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)

def getallsp(spairs):
    all_sps = []
    for spair in spairs:
        sps = spair.split("__")
        all_sps.append(sps[0])
        all_sps.append(sps[1])
    return list(set(all_sps))

def addminscore(df):
    scores = []
    S_C = [c for c in df.columns if c.startswith("Score_")]
    #for i in df.index:
    #    score = [df.loc[i,c] for c in S_C]
    #    mean, std = norm.fit(score)
    #    ci = norm.interval(0.90, loc=mean, scale=std)
    #    scores.append(ci[0])
    #df["Cutoff_Score"] = scores
    df["Cutoff_Score"] = [min([df.loc[i,c] for c in S_C]) for i in df.index]
    return df.drop(columns=S_C)

def get_pairwise_BHs(table,BHs,outdir):
    dfs = {}
    for spair,BH in BHs.items():
        sps = spair.split("__")
        s0,s1 = sps[0],sps[1]
        if s0 == s1: continue
        df = table.loc[:,[s0,s1,"Cutoff_Score"]]
        fname = os.path.join(outdir,"Cutoff_Score_{0}_{1}.tsv".format(s0,s1))
        df.to_csv(fname,header=True,index=False,sep='\t')
        dfs[(s0,s1)] = fname
    return dfs

def get_gb_mbh(BHs,outdir):
    splist,table = getallsp(BHs.keys()),None
    for i in range(len(splist)):
        for j in range(i+1,len(splist)):
            s_i,s_j = splist[i],splist[j]
            spair = "__".join(sorted([s_i,s_j]))
            bh = pd.read_csv(BHs[spair],header=0,index_col=None,sep='\t')
            table = bh if table is None else table.merge(bh)
    table.drop_duplicates(inplace=True)
    table = addminscore(table)
    tablef = os.path.join(outdir,"Global.BH")
    table.to_csv(tablef,header=True,index=False,sep='\t')
    BHs_pairwise = get_pairwise_BHs(table,BHs,outdir)
    return BHs_pairwise

class connectpairs:
    """
    Function of connecting gene pairs based on overlapped genes
    """
    def __init__(self,RBHs,tmpdir,nthreads,evalue,pep_paths,outdir):
        self.RBHs = RBHs
        self.tmpdir = tmpdir
        self.nthreads = nthreads
        self.evalue = evalue
        self.pep_paths = pep_paths
        self.outdir = outdir

    def readdf(self,fname):
        df = pd.read_csv(fname,header=0,index_col=None,sep='\t')
        return df

    def mergedf(self):
        splist = getallsp(self.RBHs.keys())
        self.splist = sorted(splist)
        N = len(self.splist)
        Df,Scores = None,[]
        for i in range(N):
            for j in range(i+1,N):
                spair = "__".join(sorted([self.splist[i],self.splist[j]]))
                df = self.readdf(self.RBHs[spair])
                column_on = self.splist[i] if i == 0 else [self.splist[i],self.splist[j]]# What if a specific species pair has no RBH result?
                Df = df if Df is None else Df.merge(df,how='outer',on=column_on)
        for r in range(Df.shape[0]):
            min_score = min([value for key,value in Df.iloc[r,:].dropna().items() if key.startswith("Score_")])
            Scores.append(min_score)
            #Scores.append(min([value for key,value in Df.loc[r,:].dropna().items() if key.startswith("Score_")]))
        Df.drop(columns=[c for c in Df.columns if c.startswith("Score_")],inplace=True)
        Df["Score"] = Scores
        self.Df = Df
        return Df

    def writedf(self,df,fname):
        df.to_csv(os.path.join(self.tmpdir,fname),header=True,index=False,sep='\t')

    def processdff(self,df,g_i,g_j,ctf):
        df_i = df[(df[0].isin([g_i,g_j])) | (df[1].isin([g_i,g_j]))]
        #df_i = pd.concat([df[df[0]==g_i],df[df[1]==g_i],df[df[0]==g_j],df[df[1]==g_j]])
        return set(df_i[df_i[13]<ctf].index)

    def processdf_getedge(self,df,g_i,g_j,ctf,G):
        df_i = pd.concat([df[df[0]==g_i],df[df[1]==g_i],df[df[0]==g_j],df[df[1]==g_j]]).drop_duplicates()
        df_ii = df_i[df_i[13]>=ctf]
        # Here under I add the possible edges between one-step neighbor
        other_genes = list(set(list(df_ii[0])+list(df_ii[1])) - set([g_i,g_j]))
        for (x, y) in itertools.product(other_genes, other_genes):
            if x==y: continue
            df_tmp = pd.concat([df[(df[0]==x) & (df[1]==y)],df[(df[1]==x) & (df[0]==y)]]).drop_duplicates()
            df_tmp = df_tmp[df_tmp[13]>=ctf]
            df_ii = pd.concat([df_ii,df_tmp])
        G.add_weighted_edges_from([(x,y,z) for x,y,z in zip(df_ii[0],df_ii[1],df_ii[13])])
        return G

    def Parallel_row(self,series,ctf):
        y = lambda x,y:"__".join(sorted([x,y]))
        N,identifier,rm_lists = len(series) - 1,[],[]
        for i in range(N):
            for j in range(i+1,N):
                identifier.append((i,j))
        for i,j in identifier:
            s_i,s_j,g_i,g_j = series.index[i],series.index[j],series[i],series[j]
            rm_lists.append((y(s_i,s_j),self.processdff(self.dmd_fs[y(s_i,s_j)],g_i,g_j,ctf)))
        return rm_lists

    def Parallel_row_whole(self,series,ctf):
        y = lambda x,y:"__".join(sorted([x,y]))
        rm_set = set()
        for i,g in enumerate(series):
            Df_reduced = self.wholedf[(self.wholedf[0]==g) | (self.wholedf[1]==g)]
            rm_set.update(set(Df_reduced[Df_reduced[13]<ctf].index))
        return rm_set

    def filterhits_pairwise(self,dmd_fs_or,postfix=".FRBH"):
        dmd_fs = {key:pd.read_csv(value,header=None,index_col=None,sep='\t',usecols=[0,1,13]) for key,value in dmd_fs_or.items()}
        RM_list = {key:set() for key in dmd_fs.keys()}
        rm_listss = Parallel(n_jobs=self.nthreads,backend='multiprocessing',batch_size=500)(delayed(self.Parallel_row)(self.Df.loc[r,:].dropna(),dmd_fs,self.Df.loc[r,:].dropna()[-1]) for r in trange(self.Df.shape[0]))
        for rm_lists in rm_listss:
            for rl in rm_lists: RM_list[rl[0]].update(rl[1])
        dmd_fs = {key:dmd_fs[key].drop(list(value)) for key,value in RM_list.items()}
        if postfix == ".FSYN":
            y = lambda x:os.path.join(*x.split("/")[:-2],"i-adhore","syntelog",os.path.basename(x))
            for key,value in dmd_fs_or.items(): dmd_fs[key].to_csv(y(value)+postfix,header=False,index=False,sep='\t')
        if postfix == ".FRBH":
            for key,value in dmd_fs_or.items(): dmd_fs[key].to_csv(value+postfix,header=False,index=False,sep='\t')

    def filterhits_pairwise_whole(self,dmd_fs_or,postfix=".FRBH"):
        if self.Df.shape[0] == 0:
            infonoseed(postfix,form='whole')
            return
        logging.info("Searching in whole")
        self.dmd_fs = {key:pd.read_csv(value,header=None,index_col=None,sep='\t',usecols=[0,1,13]) for key,value in dmd_fs_or.items()}
        self.wholedf,RM_list = pd.concat(self.dmd_fs.values()),set()
        #logging.info("Hill-climbing in step size of one tenth to find the optimal ACC")
        logging.info("Calculating the threshold that maximizes the ACC")
        newctfs = Parallel(n_jobs=self.nthreads,backend='multiprocessing')(delayed(self.get_newctfs_whole)(self.Df.loc[r,:].dropna()) for r in trange(self.Df.shape[0]))
        New_ctfs = [newctf for newctf in newctfs]
        rm_sets = Parallel(n_jobs=self.nthreads,backend='multiprocessing')(delayed(self.Parallel_row_whole)(self.Df.loc[r,:].dropna(),New_ctfs[r]) for r in trange(self.Df.shape[0]))
        for rm_set in rm_sets: RM_list.update(rm_set)
        before_drop = self.wholedf.shape[0]
        self.wholedf = self.wholedf.drop(list(RM_list))
        after_drop = self.wholedf.shape[0]
        logging.info("Drop in total {} edges".format(before_drop-after_drop))
        logging.info("{} edges remain".format(after_drop))
        if postfix == ".FSYN":
            y = lambda x:os.path.join(*x.split("/")[:-2],"i-adhore","syntelog","Syn_Concat_Score.tsv")
            self.wholedf.to_csv(y(dmd_fs_or.values[0]),header=False,index=False,sep='\t')
        if postfix == ".FRBH":
            self.wholedf.to_csv(os.path.join(self.tmpdir,"All.ABC.tsv"),header=False,index=False,sep='\t')
            self.Df["Optimal_ctf"] = New_ctfs
            self.writedf(self.Df,"Joined.RBH.Optimal")
            return os.path.join(self.tmpdir,"All.ABC.tsv")

    def filterhits_pairwise_heuristic(self,dmd_fs_or,postfix=".FRBH"):
        if self.Df.shape[0] == 0:
            infonoseed(postfix,form='pairwise')
            return
        logging.info("Searching in pairwise")
        self.dmd_fs = {key:pd.read_csv(value,header=None,index_col=None,sep='\t',usecols=[0,1,13]) for key,value in dmd_fs_or.items()}
        RM_list = {key:set() for key in self.dmd_fs.keys()}
        logging.info("First calculating the optimal preclustering cut-off per seed familiy")
        logging.info("In total {} seed families and {} threads".format(self.Df.shape[0],self.nthreads))
        newctfs = Parallel(n_jobs=self.nthreads,backend='multiprocessing')(delayed(self.get_newctfs)(self.Df.loc[r,:].dropna()) for r in trange(self.Df.shape[0]))
        New_ctfs = [newctf for newctf in newctfs]
        logging.info("Second dropping edges given the optimal preclustering cut-off per seed familiy")
        rm_listss = Parallel(n_jobs=self.nthreads,backend='multiprocessing')(delayed(self.Parallel_row)(self.Df.loc[r,:].dropna(),New_ctfs[r]) for r in trange(self.Df.shape[0]))
        for rm_lists in rm_listss:
            for spair,l in rm_lists:
                RM_list[spair].update(l)
        dmd_fs = {key:self.dmd_fs[key].drop(list(value)) for key,value in RM_list.items()}
        sum_dropped = 0
        for value in RM_list.values(): sum_dropped += len(value)
        logging.info("Dropping in total {} edges".format(sum_dropped))
        if postfix == ".FSYN":
            y = lambda x:os.path.join(*x.split("/")[:-2],"i-adhore","syntelog",os.path.basename(x))
            for key,value in dmd_fs_or.items(): dmd_fs[key].to_csv(y(value)+postfix,header=False,index=False,sep='\t')
            self.Df["Optimal_ctf"] = New_ctfs
            self.writedf(self.Df,"Syn.Net.Optimal")
        if postfix == ".FRBH":
            for key,value in dmd_fs_or.items(): dmd_fs[key].to_csv(value+postfix,header=False,index=False,sep='\t')
            self.Df["Optimal_ctf"] = New_ctfs
            self.writedf(self.Df,"Joined.RBH.Optimal")

    def get_newctfs_whole(self,series):
        ctf_boundary = series[-1]
        N = len(series)-1
        #ctf_range = (0,ctf_boundary)
        gene_list = [i for i in series[:-1]]
        #Final_ctf = self.global_ctf_per_fam_whole(N,identifier,gene_list,ctf_boundary,ctf_range)
        #Final_ctf = self.hillclimbing(gene_list,ctf_boundary,N)
        #Final_ctf1,ACC1 = self.maximizer(gene_list,ctf_boundary,N,ms=True)
        Final_ctf2,ACC2 = self.maximizer(gene_list,ctf_boundary,N,bh=True)
        return Final_ctf2
        #print((ctf_boundary,(Final_ctf1,ACC1),(Final_ctf2,ACC2)))
        #if ACC1>=ACC2: return Final_ctf1
        #else: return Final_ctf2

    def get_newctfs(self,series):
        ctf_boundary = series[-1]
        ctf_range = (0,ctf_boundary)
        gene_list,sp_list = [i for i in series[:-1]], [i for i in series.index[:-1]]
        #ctf_range = (ctf_boundary*0.5,ctf_boundary)
        N,identifier = len(series) - 1,[]
        y = lambda x,y:"__".join(sorted([x,y]))
        Ctfs = {}
        # Here I only consider local optimal ctf for each species pair, abandoned
        for i in range(N):
            for j in range(i+1,N): identifier.append((i,j))
        # Should we filter intra-specific gene pairs too?
        # Hereafter I consider a global optimal ctf per family, which might consume more CPU but less loop
        Final_ctf = self.global_ctf_per_fam(N,identifier,gene_list,sp_list,ctf_boundary,ctf_range)
        return Final_ctf

    def global_ctf_per_fam_whole(self,N,identifier,gene_list,ctf_boundary,ctf_range):
        result = minimize_scalar(lambda x: -self.calculate_property_whole(gene_list,identifier,x), bounds=ctf_range, method='bounded')
        return result.x

    def maximizer(self,gene_list,ctf,N,ms=False,bh=False):
        if ms:
            result = minimize_scalar(lambda x: -self.calACC(gene_list,x,N), bounds=(0,ctf), method='bounded')
            return result.x,result.fun
        if bh:
            x0 = [ctf]
            result = basinhopping(lambda x: -self.calACC(gene_list,x,N),x0,minimizer_kwargs={"method":"L-BFGS-B","bounds":[(0,ctf)]},niter=200)
            return result.x[0],result.fun

    def calACC(self,gene_list,ctf,N):
        new_edges = getalledges_whole(gene_list,self.wholedf,ctf,N)
        new_avg_closeness = average_closeness_centrality(edgestonet(new_edges))
        return new_avg_closeness
    
    def hillclimbing(self,gene_list,ctf,N,step=10):
        initial_hardlimit = getalledges_whole(gene_list,self.wholedf,ctf,N)
        logging.debug("Initial edges {}".format(len(initial_hardlimit)))
        assert len(initial_hardlimit) >= 1
        initial_avg_closeness = average_closeness_centrality(edgestonet(initial_hardlimit))
        best_avg_closeness,best_ctf = initial_avg_closeness,ctf
        for i in np.linspace(1-1/step,1/step,step-1):
            new_edges = getalledges_whole(gene_list,self.wholedf,i*ctf,N)
            new_avg_closeness = average_closeness_centrality(edgestonet(new_edges))
            if new_avg_closeness > initial_avg_closeness:
                best_avg_closeness,best_ctf = new_avg_closeness,i*ctf
        logging.debug("Initial cut-off {}, optimal cut-off {}".format(ctf,best_ctf))
        logging.debug("Initial ACC {}, optimal ACC {}".format(initial_avg_closeness,best_avg_closeness))
        return best_ctf

    def calculate_property_whole(self,gene_list,identifier,ctf):
        edges_list = [getalledges(gene_list[i],gene_list[j],self.wholedf,ctf) for i,j in identifier]
        Edges = []
        for edges in edges_list: Edges += edges
        G = nx.Graph()
        #G.add_weighted_edges_from([(x,y,z) for x,y,z in Edges])
        G.add_weighted_edges_from(Edges)
        CC = nx.closeness_centrality(G,distance='weight')
        ACC = sum(CC.values())/len(G)
        return ACC

    def global_ctf_per_fam(self,N,identifier,gene_list,sp_list,ctf_boundary,ctf_range):
        # First get the whole local network, not just the one-step neighbor
        result = minimize_scalar(lambda x: -self.calculate_property_pairwise(gene_list,sp_list,identifier,x), bounds=ctf_range, method='bounded')
        return result.x

    def calculate_property_pairwise(self,gene_list,sp_list,identifier,ctf):
        #edges_list = Parallel(n_jobs=self.nthreads,backend='multiprocessing')(delayed(getalledges)(gene_list[i],gene_list[j],self.dmd_fs[y(sp_list[i],sp_list[j])],ctf) for i,j in identifier)
        y = lambda x1,x2: "__".join(sorted([x1,x2]))
        edges_list = [getalledges(gene_list[i],gene_list[j],self.dmd_fs[y(sp_list[i],sp_list[j])],ctf) for i,j in identifier]
        Edges = []
        for edges in edges_list: Edges += edges
        G = nx.Graph()
        G.add_weighted_edges_from([(x,y,z) for x,y,z in Edges])
        CC = nx.closeness_centrality(G,distance='weight')
        ACC = np.mean(list(CC.values()))
        # TODO introduce a penalty of singletons
        return ACC

    def calculate_property_onepair(self,g_i,g_j,df,ctf):
        G = nx.Graph()
        G = self.processdf_getedge(df,g_i,g_j,ctf,G)
        CC = nx.closeness_centrality(G,distance='weight')
        ACC = sum(CC.values())/len(G)
        return ACC

    def calculate_property(self,series,dmd_fs,ctf):
        y = lambda x,y:"__".join(sorted([x,y]))
        N,identifier,G = len(series) - 1,[],nx.Graph()
        for i in range(N):
            for j in range(i+1,N): identifier.append((i,j))
        # Here I change it into global Dfs for just one ctf
        for i,j in identifier:
            s_i,s_j,g_i,g_j = series.index[i],series.index[j],series[i],series[j]
            G = self.processdf_getedge(dmd_fs[y(s_i,s_j)],g_i,g_j,ctf,G)
            CC = nx.closeness_centrality(G,distance='weight')
            ACC = sum(CC.values())/len(G)
        return ACC

    def buildhmm(self):
        wd = _mkdir(os.path.join(self.outdir,"HMM_references"))
        self.getneededdata(wd)
        self.concathmm = os.path.join(wd,"Concat.hmm")
        concathmm(self.jointrbh_hmm_paths,self.concathmm)
        self.run_scan()
        self.get_ctf()
        self.writedf(self.Df,"Joined.RBH.HMM")

    def get_ctf(self):
        f_g_score = {fid:{} for fid in self.famids}
        score_per_f = {fid:[] for fid in self.famids}
        cutoff_per_f = {fid:0 for fid in self.famids}
        for out in self.hmm_outfiles:
            dfo = pd.read_csv(out,header = None, index_col=False,sep ='\t')
            end = dfo.shape[0] - 10
            for i in range(3,end):
                pair = dfo.iloc[i,0].split()
                f,g,score = pair[0][:-4],pair[2],float(pair[5])
                f_g_score[f][g] = score
        for i,famid in zip(self.Df.index,self.famids):
            genes_genes = self.Df.loc[i,:].dropna().to_list()[:-1]
            for genes in genes_genes:
                for gene in genes.split(', '):
                    if f_g_score[famid].get(gene) == None:
                        logging.info("{0} has no shared domain with {1}, consider dropping this possibly false orthologue".format(gene,famid))
                        continue
                    score_per_f[famid].append(f_g_score[famid][gene])
        for famid,scores in score_per_f.items():
            if len(scores) == 0:
                logging.info("Genes in {} all have no shared domain, consider dropping this possibly false this family".format(famid))
                cutoff_per_f[famid] = 0
            else: cutoff_per_f[famid] = min(scores)*0.9
        for famid,cutoff in cutoff_per_f.items(): logging.info('The cutoff score for family {} is {:.2f}'.format(famid,cutoff))
        self.Df["HMM_score"] = [j for i,j in sorted(cutoff_per_f.items(),key=lambda x:x[0])]

    def run_scan(self):
        self.hmm_outfiles = []
        for i,j in zip(self.jointrbh_seq_paths,self.jointrbh_hmm_paths):
            cmd1 = ['hmmpress'] + [j]
            sp.run(cmd1, stdout=sp.PIPE,stderr=sp.PIPE)
            cmd2 = ['hmmscan','--tblout','{}.tbl'.format(i[:-4]),'--noali','-E',str(self.evalue),j,i]
            sp.run(cmd2, stdout=sp.PIPE,stderr=sp.PIPE)
            self.hmm_outfiles.append('{}.tbl'.format(i[:-4]))

    def getneededdata(self,wd):
        self.allneededseq()
        self.writepepperfam(wd)
        self.maffthmmbuild()

    def maffthmmbuild(self):
        self.jointrbh_mafft_paths = [i+'.aln' for i in self.jointrbh_seq_paths]
        self.jointrbh_hmm_paths = [i+'.aln.hmm' for i in self.jointrbh_seq_paths]
        for i,j,k in zip(self.jointrbh_seq_paths,self.jointrbh_mafft_paths,self.jointrbh_hmm_paths):
            mafft_cmd(i,'--auto',j)
            cmdhmmbuild(j,k)

    def allneededseq(self):
        self.seqs_dit = {}
        y = lambda x: [i for i in self.pep_paths if os.path.basename(i) == x][0]
        for pep_path in self.Df.columns[:-1]:
            genes = self.Df[pep_path].dropna().to_list()
            if len(genes) == 0: continue
            self.seqs_dit.update({record.id:record.seq for record in SeqIO.parse(y(pep_path), 'fasta') if record.id in genes})

    def writepepperfam(self,wd):
        x = lambda i : "GF{:0>8}.pep".format(i+1)
        y = lambda x:os.path.join(wd,x)
        self.jointrbh_seq_paths = []
        self.famids = []
        for i in range(self.Df.shape[0]):
            famid = x(i);self.famids.append(famid[:-4])
            jointrbh_seq_path = y(famid);self.jointrbh_seq_paths.append(jointrbh_seq_path)
            with open(jointrbh_seq_path,'w') as f:
                for gene in self.Df.iloc[i,:-1].dropna(): f.write(">{}\n{}\n".format(gene,self.seqs_dit[gene]))

def simple_df(f):
    pd.read_csv(f,header=None,index_col=None,sep='\t',usecols=[0,1,13]).to_csv(f+'_tmp',header=False,index=False,sep='\t')
    return f+'_tmp'

def Catalldmd(fs,outf):
    f_simples = [simple_df(f) for f in fs]
    cmd = ["cat"] + [f for f in f_simples]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    with open(outf, 'w') as f: f.write(out.stdout.decode('utf-8'))
    cmd = ["rm"] + [f for f in f_simples]
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

def precluster_rbhfilter(RBHs,dmd_pairwise_outfiles,outdir,nthreads,evalue,pep_paths):
    tmpdir = _mkdir(os.path.join(outdir,"MCL"))
    logging.info("Connecting RBH seed orthologues")
    ConnectPair = connectpairs(RBHs,tmpdir,nthreads,evalue,pep_paths,outdir)
    ConnectPair.mergedf()
    ConnectPair.writedf(ConnectPair.Df,"Joined.RBH")
    logging.info("Discarding low-similarity edges")
    #concatf = ConnectPair.filterhits_pairwise_whole(dmd_pairwise_outfiles)
    ConnectPair.buildhmm()
    return concatf
    #ConnectPair.filterhits_pairwise(dmd_pairwise_outfiles)
    #ConnectPair.filterhits_pairwise_heuristic(dmd_pairwise_outfiles)

def precluster_bhfilter(BHs,dmd_pairwise_outfiles,tmpdir,nthreads):
    BHs_pairwise = get_gb_mbh(BHs,tmpdir)
    y = lambda x: os.path.join(tmpdir,"__".join(sorted([x[0],x[1]]))+".RBH")
    y0 = lambda x: os.path.join(tmpdir,"__".join(sorted([x[0],x[0]]))+".RBH")
    y1 = lambda x: os.path.join(tmpdir,"__".join(sorted([x[1],x[1]]))+".RBH")
    dmd_pairwise_outfiles_BH = {}
    logging.info("Cutting edges based on seed orthologs")
    for spair,BH in BHs_pairwise.items():
        logging.info("{0} vs. {1}".format(sorted([spair[0],spair[1]])[0],sorted([spair[0],spair[1]])[1]))
        gxy,ctf = Processor_DMD(BH).get_rbh_score()
        dmd_handle = DMD_ABC(dmd_pairwise_outfiles['__'.join(sorted([spair[0],spair[1]]))],gxy,nthreads,y(spair),ctf=ctf)
        dmd_handle.filterdf()
        dmd_handle = DMD_ABC(dmd_pairwise_outfiles['__'.join(sorted([spair[0],spair[0]]))],gxy,nthreads,y0(spair),ctf=ctf)
        dmd_handle.filterdf()
        dmd_handle = DMD_ABC(dmd_pairwise_outfiles['__'.join(sorted([spair[1],spair[1]]))],gxy,nthreads,y1(spair),ctf=ctf)
        dmd_handle.filterdf()
        dmd_pairwise_outfiles_BH.update({'__'.join([spair[0],spair[1]]):y(spair),'__'.join(sorted([spair[0],spair[0]])):y0(spair),'__'.join(sorted([spair[1],spair[1]])):y1(spair)})
    return dmd_pairwise_outfiles_BH

def precluster_apfilter(aps,dmd_pairwise_outfiles,outdir,nthreads):
    y = lambda x: os.path.join(outdir,"__".join(sorted([x[0],x[1]]))+".AP")
    y0 = lambda x: os.path.join(outdir,"__".join(sorted([x[0],x[0]]))+".AP")
    y1 = lambda x: os.path.join(outdir,"__".join(sorted([x[1],x[1]]))+".AP")
    dmd_pairwise_outfiles_AP = {}
    for spair,fn in aps.items():
        logging.info("Filtering hits based on collinearity between {0} and {1}".format(sorted([spair[0],spair[1]])[0],sorted([spair[0],spair[1]])[1]))
        df = pd.read_csv(fn,sep="\t",index_col=0)
        if len(df.columns) == 5: df = df.drop(columns=['gene_y']).rename(columns = {'gene_x':'gene_y'}).rename(columns = {'Unnamed: 2':'gene_x'})
        df['gene_xy'] = ["__".join(sorted([gx,gy])) for gx,gy in zip(df['gene_x'],df['gene_y'])]
        df = df.drop_duplicates(subset=['gene_xy'])
        gxy = [(gx,gy) for gx,gy in zip(df['gene_x'],df['gene_y'])]
        dmd_handle = DMD_ABC(dmd_pairwise_outfiles['__'.join(sorted([spair[0],spair[1]]))],gxy,nthreads,y(spair))
        ctf = dmd_handle.filterdf_getctf()
        dmd_handle = DMD_ABC(dmd_pairwise_outfiles['__'.join(sorted([spair[0],spair[0]]))],gxy,nthreads,y0(spair),ctf=ctf)
        dmd_handle.filterdf()
        dmd_handle = DMD_ABC(dmd_pairwise_outfiles['__'.join(sorted([spair[1],spair[1]]))],gxy,nthreads,y1(spair),ctf=ctf)
        dmd_handle.filterdf()
        dmd_pairwise_outfiles_AP.update({'__'.join([spair[0],spair[1]]):y(spair),'__'.join(sorted([spair[0],spair[0]])):y0(spair),'__'.join(sorted([spair[1],spair[1]])):y1(spair)})
    return dmd_pairwise_outfiles_AP

def concatAPs_ABC(dic):
    cmd = ["cat"] + [fn for fn in dic.values()]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    with open(faln, 'w') as f: f.write(out.stdout.decode('utf-8'))

def filter_interfamedge_getctf(gxy,df,nthreads):
    index_cfs = Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11,batch_size=500)(delayed(getrmindex_getctf)(gx,gy,df) for gx,gy in gxy)
    index_torms,cfs = (i[0] for i in index_cfs),(j[1] for j in index_cfs)
    return df.drop(list(chain(*index_torms))),dict(ChainMap(*cfs))

def filter_interfamedge(gxy,df,ctf,nthreads):
    index_torms = Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=500)(delayed(getrmindex)(gx,gy,df,ctf) for gx,gy in tqdm(gxy,unit="item"))
    return df.drop(list(chain(*index_torms)))

def getrmindex_getctf(gx,gy,df):
    index_torm,cf = [],{}
    df_xy = df[(df[0]==gx)&(df[1]==gy)]
    if len(df_xy) == 0:
        df_xy = df[(df[1]==gx)&(df[0]==gy)]
        cutoff_score = list(df_xy[13])[0]
        cf[(gx,gy)] = cutoff_score
        index_torm += list(df[(df[1]==gx)&(df[13]<cutoff_score)].index)
        index_torm += list(df[(df[0]==gy)&(df[13]<cutoff_score)].index)
    else:
        cutoff_score = list(df_xy[13])[0]
        cf[(gx,gy)] = cutoff_score
        index_torm += list(df[(df[0]==gx)&(df[13]<cutoff_score)].index)
        index_torm += list(df[(df[1]==gy)&(df[13]<cutoff_score)].index)
    return (index_torm,cf)

def getrmindex(gx,gy,df,ctf):
    index_torm = []
    index_torm += list(df[(df[0]==gx)&(df[13]<ctf[(gx,gy)])].index)
    index_torm += list(df[(df[1]==gx)&(df[13]<ctf[(gx,gy)])].index)
    index_torm += list(df[(df[0]==gy)&(df[13]<ctf[(gx,gy)])].index)
    index_torm += list(df[(df[1]==gy)&(df[13]<ctf[(gx,gy)])].index)
    return index_torm

def mergefirst(first,value,final_list):
    value.remove(first)
    for g1,g2 in value:
        if g1 in first or g2 in first:
            first = list(set([g for g in first] + [g1,g2]))
            value.remove((g1,g2))
    final_list.append(first)
    if len(value) <= 1:
        return final_list
    else:
        return mergefirst(value[0],value,final_list)

def _label_families(df):
    df.index = ["GF{:0>8}".format(i+1) for i in range(len(df.index))]

def writeseedog(pair,final_list,outdir,gsmap):
    OGs = []
    for og in final_list:
        OG = {sp:"" for sp in set(gsmap.values())}
        for g in og:
            s = gsmap[g]
            OG[s] = g if OG[s]=="" else ", ".join([OG[s],g])
        OGs.append(OG)
    df = pd.DataFrame.from_dict(OGs)
    _label_families(df)
    fname = os.path.join(outdir,"{0}_{1}_Seed_SynFam.tsv".format(pair[0],pair[1]))
    df.to_csv(fname,header=True,index=True,sep='\t')
    logging.info("The path is {}".format(fname))
    return fname

def mergeso(value):
    final_list = mergefirst(value[0],value,[])
    return final_list

def synseedortho(aps,outdir,gsmap):
    Seed_Orthos,OA_Orthos,aplist = {key:[] for key in aps.keys()},[],[]
    for key,value in aps.items():
        df = pd.read_csv(value,sep="\t",index_col=0)
        if len(df.columns) == 5: df = df.drop(columns=['gene_y']).rename(columns = {'gene_x':'gene_y'}).rename(columns = {'Unnamed: 2':'gene_x'})
        df['gene_xy'] = ["__".join(sorted([gx,gy])) for gx,gy in zip(df['gene_x'],df['gene_y'])]
        df = df.drop_duplicates(subset=['gene_xy'])
        if key[0]!=key[1]: aplist += list(df['gene_xy'])
        OA_Orthos += [(gx,gy) for gx,gy in zip(df['gene_x'],df['gene_y'])]
    OA_Orthos = mergeso(OA_Orthos)
    logging.info("Writing seed syntenic orthofamilies")
    seedf = writeseedog(("Ortho","Inpara"),OA_Orthos,outdir,gsmap)
    return seedf,aplist

def getallgs(d):
    Gs = []
    for gs in d:
        for g in gs.split(", "):
            Gs.append(g)
    return Gs

def getallgsdf(df,g,cf,gsmap):
    Gs = [i for i,s in zip(df[0],df[13]) if i!=g and s>=cf] + [i for i,s in zip(df[1],df[13]) if i!=g and s>=cf]
    Gs = [(i,gsmap[i]) for i in Gs]
    return Gs

def multi_adhomo(i,seedf,pair_bit,aplist,gsmap,df):
    orthoss = []
    d = seedf.loc[i,:].dropna()
    cutoff = min(pair_bit["__".join(sorted([x,y]))] for x,y in itertools.product(*(j.split(", ") for j in d)) if "__".join(sorted([x,y])) in aplist)
    for g in getallgs(d): orthoss.append((i,getallgsdf(pd.concat([df[df[0]==g],df[df[1]==g]]),g,cutoff,gsmap)))
    return (orthoss,(i,cutoff))

def oadd(iter_orthoss_cutoff,seedf):
    for orthoss,_ in iter_orthoss_cutoff:
        if orthoss is None: continue
        for i,orthos in orthoss:
            for ortho,sp in orthos:
                if ortho in chain_iter(p.split(', ') for p in seedf[sp].dropna()): continue
                seedf.loc[i,sp] = ", ".join([seedf.loc[i,sp],ortho])
    return seedf

def adhomosf(df,seedf,gsmap,aplist,nthreads):
    pair_bit = {p:s for p,s in zip(df[14],df[13])}
    cutoffs,yy = {},lambda x: (x,gsmap[x])
    good_indexs = [i for i in seedf.index if len(seedf.loc[i,:].dropna())>1]
    iter_orthoss_cutoff = Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11,batch_size=500)(delayed(multi_adhomo)(i,seedf,pair_bit,aplist,gsmap,df) for i in good_indexs)
    cutoffs = {j[0]:j[1] for i,j in iter_orthoss_cutoff if not (j[0] is None)}
    seedf = oadd(iter_orthoss_cutoff,seedf)
    return seedf

def chain_iter(item):
    for i in item:
        for j in i:
            yield j

def addxy(df):
    df[14] = ["__".join(sorted([g1,g2])) for g1,g2 in zip(df[0],df[1])]
    return df

def addortho(seedfn,dmd_pairwise_outfiles,gsmap,outdir,aplist,nthreads):
    seedf = pd.read_csv(seedfn,header=0,index_col=0,sep='\t')
    logging.info("Initial expanding syntenic orthofamilies using {} threads".format(nthreads))
    yy = lambda x:pd.read_csv(x,header=None,index_col=None,sep='\t',usecols=[0,1,13])
    score_df = pd.concat([addxy(yy(fn)) for fn in dmd_pairwise_outfiles.values()]).sort_values(by=[13],ascending=False)
    addhomoseedf = adhomosf(score_df,seedf,gsmap,aplist,nthreads)
    fname = os.path.join(outdir,"Ortho_Inpara_Seed_SynFam_IniExpand.tsv")
    addhomoseedf.to_csv(fname,header=True,index=True,sep='\t')
    logging.info("The path is {}".format(fname))
    return fname

def mafft_cmd(fpep,o,fpaln):
    cmd = ["mafft"] + o.split() + ["--amino", fpep]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    with open(fpaln, 'w') as f: f.write(out.stdout.decode('utf-8'))

def run_hmmerbp(ids,fp,Seqs):
    with open(fp,'w') as f:
        for i in ids: f.write('>{0}\n{1}\n'.format(i,Seqs[i]))
    mafft_cmd(fp,'--auto',fp+'.aln')
    cmd = ['hmmbuild'] + [fp+'.hmm'] + [fp+'.aln']
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def getgt(*args):
    S = set()
    for arg in args:
        for g in arg: S.add(g)
    return S

def set2seq(All_pep_IDs,pep_paths):
    Seqs = {}
    for pep_path in pep_paths:
        for record in SeqIO.parse(pep_path, 'fasta'):
            if record.id in All_pep_IDs: Seqs[record.id] = record.seq
    return Seqs

def run_scan(hmmf,pep_path,evalue):
    pf = pep_path[:-4]
    cmd = ['hmmscan','--tblout','{}.tbl'.format(pf),'--noali','-E',str(evalue),hmmf,pep_path]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def allscores(outs,fams):
    f_g_score,minor_gsmap = {f:{} for f in fams},{}
    for out in outs:
        dfo = pd.read_csv(out,header = None, index_col=False,sep ='\t')
        addedgs = {}
        for i in range(3,dfo.shape[0]-10):
            pair = dfo.iloc[i,0].split()
            f,g,score = pair[0][:-4],pair[2],float(pair[5])
            if g in addedgs:
                if addedgs[g][0] < score:
                    del f_g_score[addedgs[g][1]][addedgs[g][2]]
                    f_g_score[f][g],addedgs[g],minor_gsmap[g] = score,(score,f,g),os.path.basename(out)[:-4]
            else: f_g_score[f][g],addedgs[g],minor_gsmap[g] = score,(score,f,g),os.path.basename(out)[:-4]
    return f_g_score,minor_gsmap

def domain_add(seedf,pep_paths,outdir):
    outs = (i[:-4]+'.tbl' for i in pep_paths)
    Scores,gsmap = allscores(outs,seedf.index)
    cf = getreferscore(Scores,seedf)
    seedf = add_by_cf(cf,Scores,seedf,gsmap)

def add_by_cf(cf,Scores,seedf,gsmap):
    for fam,scores in Scores.items():
        for g,s in scores.items():
            if s >= cf[fam]:
                sp = gsmap[g]
                if type(seedf.loc[fam,sp]) is float:
                    seedf.loc[fam,sp] = g
                else:
                    seedf.loc[fam,sp] = seedf.loc[fam,sp] + ", {}".format(g)
    return seedf

def getreferscore(Scores,seedf):
    scores,cf = {},{}
    for i in seedf.index:
        scores[i] = set()
        ids = ', '.join(list(seedf.loc[i,:].dropna())).split(', ')
        for g in ids:
            if Scores[i].get(g,None) is None:
                logging.info("{0} doesn't share domain with remaining members in {1}".format(g,i))
                continue
            scores[i].add([Scores[i][g]])
    for k,v in scores.items():
        if len(v) == 0:
            logging.info("No domain was found among seed orthologues in {}".format(k))
            cf[k] = 0
            continue
        cf[k] = min(v) * 0.9
    return cf

def concathmm(hmmfs,fn):
    cmd = ['cat'] + [i for i in hmmfs]
    out = sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    with open(fn,'w') as f: f.write(out.stdout.decode('utf-8'))
    cmd = ['hmmpress'] + [fn]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def buildrefhmm(df,pep_paths,nthreads,outdir):
    main_wd = _mkdir(os.path.join(outdir,"hmmer_results"))
    refer_wd = _mkdir(os.path.join(main_wd,"hmm_references"))
    yids = lambda i: ', '.join(list(df.loc[i,:].dropna())).split(', ')
    yfnp = lambda i: os.path.join(refer_wd,'{}.pep'.format(i))
    All_pep_IDs = getgt(*(yids(i) for i in df.index))
    All_pep_Seqs = set2seq(All_pep_IDs,pep_paths)
    Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11,batch_size=500)(delayed(run_hmmerbp)(yids(i),yfnp(i),All_pep_Seqs) for i in df.index)
    hmmf = os.path.join(refer_wd,'Full.hmm')
    concathmm((yfnp(i)+'.hmm' for i in df.index),hmmf)
    return hmmf

def addorthohmm(seedfn,pep_paths,nthreads,outdir,evalue):
    seedf = pd.read_csv(seedfn,header=0,index_col=0,sep='\t')
    hmmf = buildrefhmm(seedf,pep_paths,nthreads,outdir)
    Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(run_scan)(hmmf,pep_path,evalue) for pep_path in pep_paths)
    seedf = domain_add(seedf,pep_paths,outdir)
    fname = os.path.join(outdir,'Ortho_Inpara_Seed_SynFam_SecExpand.tsv')
    seedf.to_csv(fname,header=True,index=True,sep='\t')
    logging.info("The path is {}".format(fname))

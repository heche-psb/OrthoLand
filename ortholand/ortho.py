import os
import uuid
import logging
from Bio import SeqIO
import subprocess as sp
import pandas as pd
from scipy import stats
import numpy as np
import sys
import itertools

sys.setrecursionlimit(10000000)

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
    gsmap,gldict = {},{}
    with open(fname,'w') as f:
        for record in SeqIO.parse(fn, 'fasta'):
            if not (gsmap.get(record.id) is None):
                logging.error("Duplicated gene id found in {}".format(os.path.basename(fn)))
                exit(1)
            gsmap[record.id] = os.path.basename(fn)
            aa_seq = record.translate(to_stop=to_stop, cds=cds, id=record.id)
            gldict[record.id] = len(aa_seq)
            f.write(">{0}\n{1}\n".format(record.id,aa_seq.seq))
    return fname,gsmap,gldict

def writepep(data,tmpdir,outdir,to_stop,cds):
    logging.info("Translating cds to pep")
    parent,pep_paths,gsmaps,gldicts = os.getcwd(),[],{},{}
    if tmpdir is None: tmpdir = "tmp_" + str(uuid.uuid4())
    _mkdir(tmpdir)
    fnames_seq = listdir(data)
    for fn in fnames_seq:
        pep_path,gsmap,gldict = writepepfile(fn,tmpdir,to_stop,cds)
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

def pairwise_diamond(pep_paths,evalue,nthreads,outdir,gldict):
    pep_path_dbs,outfiles = [i+'.dmnd' for i in pep_paths],{}
    logging.info("Running diamond and normalization")
    for pep_path in pep_paths: mkdb(pep_path,nthreads)
    for i,pep_path in enumerate(pep_paths):
        for j in range(i,len(pep_paths)):
            pep_path_db = pep_path_dbs[j]
            s1,s2=os.path.basename(pep_path),os.path.basename(pep_path_db)[:-5]
            outfiles["__".join(sorted([s1,s2]))]=pairdiamond(pep_path,pep_path_db,nthreads,evalue,outdir,gldict)
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
    pep_paths,tmppath,gsmap,gldict = writepep(data,tmpdir,outdir,to_stop,cds)
    dmd_pairwise_outfiles = pairwise_diamond(pep_paths,evalue,nthreads,outdir,gldict)
    logging.info("Diamond done")
    rmalltmp(tmppath)
    return gsmap,dmd_pairwise_outfiles

def syninfer(gsmap,dmd_pairwise_outfiles,parameters,gff3s,features,attributes,outdir,config=None):
    aps=pairwise_iadhore(gsmap,dmd_pairwise_outfiles,parameters,gff3s,features,attributes,outdir,config=config)
    return aps

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
    para_dict = {"gap_size":30,"q_value":0.75,"cluster_gap":35,"prob_cutoff":0.01,"anchor_points":3,"alignment_method":"gg2","level_2_only":"false","multiple_hypothesis_correction":"FDR","visualizeGHM":"false","visualizeAlignment":"false"}
    if not (parameters is None):
        # "gap_size=30;q_value=0.75"
        for para in parameters.split(";"):
            key,value = para.split("=")
            if para_dict.get(key.strip()) is None:
                logging.error("The parameter {} is not included in i-adhore! Plesae double check".format(key.strip()))
            else: para_dict[key.strip()]=value.strip()
    with open(fname,"w") as f:
        for key,value in para_dict.items(): f.write(key+"="+str(value)+"\n")
        f.write("genome={}\n".format(sp_i))
        for scaf,path in gene_lists_i.items(): f.write(scaf+" "+path+"\n")
        f.write("\n")
        f.write("genome={}\n".format(sp_j))
        for scaf,path in gene_lists_j.items(): f.write(scaf+" "+path+"\n")
        f.write("\n")
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
    logging.info("Writing genelists for i-adhore")
    for gff3,feature,attribute in zip(gff3s,features,attributes):
        splist_dir = _mkdir(os.path.join(main_wd,os.path.basename(gff3)+"_genelists"))
        gff_info,sp = gff2table(gff3,feature,attribute,gsmap)
        sps[os.path.basename(gff3)] = sp
        gff_infos[os.path.basename(gff3)] = gff_info
        gene_lists[os.path.basename(gff3)] = writegenelist(gff_info,splist_dir)
    logging.info("Running i-adhore")
    aps = {}
    for i in range(len(gff3s)):
        for j in range(i,len(gff3s)):
            key_i,key_j = os.path.basename(gff3s[i]),os.path.basename(gff3s[j])
            sp_i,sp_j,gene_lists_i,gene_lists_j = sps[key_i],sps[key_j],gene_lists[key_i],gene_lists[key_j]
            logging.info("{0} vs. {1}".format(sp_i,sp_j))
            dirname = _mkdir(os.path.join(main_wd,"__".join(sorted([sp_i,sp_j]))))
            pathbt = writeblastable(dmd_pairwise_outfiles["__".join(sorted([sp_i,sp_j]))],os.path.join(dirname,"blast_table.txt"))
            fconf = writepair_iadhoreconf(sp_i,sp_j,gene_lists_i,gene_lists_j,parameters,dirname,pathbt)
            run_adhore(fconf)
            #aps["__".join(sorted([sp_i,sp_j]))] = os.path.join(dirname,"iadhore-out","anchorpoints.txt")
            aps[(sp_i,sp_j)] = os.path.join(dirname,"iadhore-out","multiplicon_pairs.txt")
    logging.info("I-adhore done")
    return aps

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
        #df = pd.read_csv(value,header=0,index_col=0,sep='\t')
        df = pd.read_csv(value,sep="\t",index_col=0)
        if len(df.columns) == 5: df = df.drop(columns=['gene_y']).rename(columns = {'gene_x':'gene_y'}).rename(columns = {'Unnamed: 2':'gene_x'})
        df['gene_xy'] = ["__".join(sorted([gx,gy])) for gx,gy in zip(df['gene_x'],df['gene_y'])]
        df = df.drop_duplicates(subset=['gene_xy'])
        if key[0]!=key[1]: aplist += list(df['gene_xy'])
        OA_Orthos += [(gx,gy) for gx,gy in zip(df['gene_x'],df['gene_y'])]
        #for gx,gy in zip(df['gene_x'],df['gene_y']): Seed_Orthos[key].append((gx,gy)) # in case "__" in original gene id
    #Seed_Orthos = {key:mergeso(value) for key,value in Seed_Orthos.items()}
    OA_Orthos = mergeso(OA_Orthos)
    logging.info("Writing seed syntenic orthofamilies")
    seedf = writeseedog(("Ortho","Inpara"),OA_Orthos,outdir,gsmap)
    return seedf,aplist
    #for key,value in Seed_Orthos.items():
    #    if key[0] != key[1]: writeseedog(key,value,outdir,gsmap)

#def besthitcf(sp,gs,pair_bit):
#    for gs

def getallgs(d):
    Gs = []
    for gs in d:
        for g in gs.split(", "):
            Gs.append(g)
    return Gs

def adhomosf(pair_bit,seedf,gsmap,aplist):
    cutoffs = {}
    y = lambda x,z: next(i for i in x if i!=z)
    yy = lambda x: (x,gsmap[x])
    for i in seedf.index:
        d = seedf.loc[i,:].dropna()
        if len(d) == 1:
            continue
        cutoff = min([pair_bit["__".join(sorted([x,y]))] for x,y in itertools.product(*(j.split(", ") for j in d)) if "__".join(sorted([x,y])) in aplist])
        #scores = []
        #for x,y in itertools.product(*(j.split(", ") for j in d)):
        #    if "__".join(sorted([x,y])) not in aplist: continue
        #    score = pair_bit.get("__".join(sorted([x,y])),0)
        #    if score == 0:
        #        logging.info("Can't find score for {0} in {1}".format("__".join(sorted([x,y])),i))
        #        continue
        #    scores.append(score)
        #cutoff = min(scores)
        logging.info("cutoff for {0} is {1:.5f}".format(i,cutoff))
        #cutoff = set((pair_bit.get("__".join(sorted([x,y])),0) for x,y in itertools.product(*(j.split(", ") for j in d))))
        #if 0 in cutoff: cutoff.remove(0)
        #if len(cutoff) == 0:
        #    logging.info("Can't find diamond score in {}".format(i))
        #    continue
        #cutoff = min(cutoff)
        cutoffs[i] = cutoff
        for g in getallgs(d):
            #orthos = (yy(y(key.split("__"),g)) for key,value in pair_bit.items() if g in key and value >= cutoff and g+"__"+g!=key)
            orthos = (yy(key.replace(g,'').replace('__','')) for key,value in pair_bit.items() if g in key and value >= cutoff and g+"__"+g!=key)
            for ortho,sp in orthos:
                if ortho in chain_iter(p.split(', ') for p in seedf[sp].dropna()): continue
                #if ortho in seedf.loc[i,sp]: continue
                seedf.loc[i,sp] = ", ".join([seedf.loc[i,sp],ortho])
    return seedf

def chain_iter(item):
    for i in item:
        for j in i:
            yield j

def addortho(seedfn,dmd_pairwise_outfiles,gsmap,outdir,aplist):
    seedf = pd.read_csv(seedfn,header=0,index_col=0,sep='\t')
    logging.info("Initial expanding syntenic orthofamilies")
    pair_bit, y = {}, lambda x:(x[0],x[1],x[13])
    for fn in dmd_pairwise_outfiles.values():
        df = pd.read_csv(fn,header=None,index_col=None,sep='\t')
        pair_bit.update({"__".join(sorted([g1,g2])):s for g1,g2,s in zip(df[0],df[1],df[13])})
        #pair_bit.update({"__".join(sorted([g1,g2])):s for g1,g2,s in zip(*y(pd.read_csv(fn,header=None,index_col=None,sep='\t')))})
    addhomoseedf = adhomosf(pair_bit,seedf,gsmap,aplist)
    fname = os.path.join(outdir,"Ortho_Inpara_Seed_SynFam_IniExpand.tsv")
    addhomoseedf.to_csv(fname,header=True,index=True,sep='\t')
    logging.info("The path is {}".format(fname))


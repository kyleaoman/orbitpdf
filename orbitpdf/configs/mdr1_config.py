import numpy as np

#---------------DON'T TOUCH THIS-------------------
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

cfg = AttrDict()
pdf_cfg = AttrDict()
#--------------------------------------------------

cfg.h0 = .7 #MDR1 = .7, Bolshoi = .7, DOVE = .704
cfg.m_min_cluster = 1.E13 #Msun (not h^-1)
cfg.m_max_cluster = 1.E30 #Msun (not h^-1)
cfg.m_min_satellite = 1.E0 #Msun (not h^-1)
cfg.m_max_satellite = 1.E30 #Msun (not h^-1)
cfg.lbox = 1420.0 #Mpc (not h^-1); MDR1 = 1420.0 (=1 h^-1 Gpc), Bolshoi = 355.11 (=250 h^-1 Mpc), DOVE = 100.0
cfg.interloper_dR = 2.5 #in units of cluster virial radius
cfg.interloper_dV = 2.0 #in units of cluster velocity dispersion
cfg.outfile = './mdr1_orbits.hdf5' #output file
cfg.treedir = '/home/kaoman/CT_Multidark_Output/tree/' #location of consistent-trees tree files
cfg.scalefile = './mdr1_scales.txt' #list of simulation output numbers (col 1) scale factors (col 2)
cfg.skipsnaps = 0 #skip snapshots at end to make high-z catalog
cfg.ncpu = 7

#PARAMETERS BELOW FOR PDFS.PY ONLY
pdf_cfg.resolution_cut = 7.9432E11 #Msun (not h^-1); MDR1 = 10^11.9, Bolshoi=10^11.9/64, bolshoi particle is 1.93E8 Msun (physical), dove particle is 8.8E6 Msun (physical) -> DOVE = 10^8.75
pdf_cfg.rbins = np.linspace(0, 2.5, 101) #min, max, nbins+1
pdf_cfg.vbins = np.linspace(0, 2.0, 101) #min, max, nbins+1
pdf_cfg.pdf_m_min_satellite = 1.E9
pdf_cfg.pdf_m_max_satellite = 1.E10
pdf_cfg.pdf_m_min_cluster = cfg.m_min_cluster
pdf_cfg.pdf_m_max_cluster = cfg.m_max_cluster
pdf_cfg.pdfsfile = './void.hdf5'

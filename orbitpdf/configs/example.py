import numpy as np
import astropy.units as U
from orbitpdf import Orbits, OrbitsConfig, OrbitPDFConfig, InfallTimeOrbitPDF

cfg = OrbitsConfig()

cfg.h0 = .7 #MDR1 = .7, Bolshoi = .7, DOVE = .704
cfg.m_min_cluster = 1.E13 * U.Msun #Msun (not h^-1)
cfg.m_max_cluster = 1.E30 * U.Msun #Msun (not h^-1)
cfg.m_min_satellite = 1.E0 * U.Msun #Msun (not h^-1)
cfg.m_max_satellite = 1.E30 * U.Msun #Msun (not h^-1)
cfg.lbox = 355.11 * U.Mpc #Mpc (not h^-1); MDR1 = 1420.0 (=1 h^-1 Gpc), Bolshoi = 355.11 (=250 h^-1 Mpc), DOVE = 100.0
cfg.interloper_dR = 2.5 #in units of cluster virial radius
cfg.interloper_dV = 2.0 #in units of cluster velocity dispersion
cfg.treedir = '/home/kaoman/CT_Bolshoi_Output/tree/' #location of consistent-trees tree files
cfg.scalefile = '/home/kaoman/code/orbitpdf/orbitpdf/configs/bolshoi_scales.txt' #list of simulation output numbers (col 1) scale factors (col 2)
cfg.skipsnaps = 0 #skip snapshots at end to make high-z catalog
cfg.ncpu = 2 #max cpus to use for parallel code segments
cfg.outfile = './testout.hdf5'

orb = Orbits(cfg=cfg)
#UNCOMMENT lines below to actually run search
#orb.cluster_search()
#orb.interloper_search()
#orb.orbit_search()

pdfcfg = OrbitPDFConfig(cfg) #copies config above, some values, e.g. lbox, h0, will be re-used
pdfcfg.orbitfile = 'bolshoi_orbits.hdf5' #cfg.outfile
pdfcfg.resolution_cut = 1.2411E10 * U.Msun #Msun (not h^-1); MDR1 = 10^11.9, Bolshoi=10^11.9/64, bolshoi particle is 1.93E8 Msun (physical), dove particle is 8.8E6 Msun (physical) -> DOVE = 10^8.75
pdfcfg.rbins = np.linspace(0, 2.5, 101) #min, max, nbins+1
pdfcfg.vbins = np.linspace(0, 2.0, 101) #min, max, nbins+1
pdfcfg.pdf_m_min_satellite = np.power(10, [9., 9.5, 10., 10.5, 11.]) * U.Msun #lower bin edges for satellite mass
pdfcfg.pdf_m_max_satellite = np.power(10, [9.5, 10., 10.5, 11., 11.5]) * U.Msun #upper bin edges for satellite mass
pdfcfg.pdf_m_min_cluster = np.power(10, [13., 14.]) * U.Msun #lower bin edges for cluster mass
pdfcfg.pdf_m_max_cluster = np.power(10, [14., 15.]) * U.Msun #upper bin edges for cluster mass
pdfcfg.pdfsfile = './testpdfsout.hdf5'

pdf = InfallTimeOrbitPDF(cfg=pdfcfg)
#UNCOMMENT lines below to actually generate pdf
#pdf.process_orbits()
#pdf.calculate_pdfs()
#pdf.write()

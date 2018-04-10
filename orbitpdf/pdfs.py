import h5py
import numpy as np
from sys import argv
from itertools import product
from datetime import datetime
if len(argv) < 2:
    print 'Provide config file as argument.'
    exit()
execfile(argv[1])

def timer():
    T = datetime.now()
    return ' [{0:02d}:{1:02d}:{2:02d}]'.format(T.hour, T.minute, T.second)

def delta_RV(sat, cluster):
    rel_xyz = sat['xyz'][-1] - cluster['xyz'][-1]
    rel_xyz[rel_xyz < cfg.lbox / 2.] += cfg.lbox
    rel_xyz[rel_xyz > cfg.lbox / 2.] -= cfg.lbox
    return np.sqrt(np.sum(np.power(rel_xyz[:2], 2))) / (1.E-3 * cluster['rvir'][-1]), np.abs(sat['vxyz'][-1, 2] - cluster['vxyz'][-1, 2] + 100.0 * cfg.h0 * rel_xyz[2]) / cluster['vrms'][-1]

def delta_RV_interlopers(cluster):
    rel_xyz = cluster['interlopers/xyz'] - cluster['xyz'][-1]
    rel_xyz[rel_xyz < cfg.lbox / 2.] += cfg.lbox
    rel_xyz[rel_xyz > cfg.lbox / 2.] -= cfg.lbox
    return np.sqrt(np.sum(np.power(rel_xyz[:, :2], 2), axis=1)) / (1.E-3 * cluster['rvir'][-1]), np.abs(cluster['interlopers/vxyz'][:, 2] - cluster['vxyz'][-1, 2] + 100.0 * cfg.h0 * rel_xyz[:, 2]) / cluster['vrms'][-1]
    

rbins = pdf_cfg.rbins
vbins = pdf_cfg.vbins
Nsatbins = len(pdf_cfg.pdf_m_min_satellite)
Nclusterbins = len(pdf_cfg.pdf_m_min_cluster)

statistics = {
    'clustermass': 0,
    'satmass': 0,
    'preaccretion': 0,
    'accretionmass': 0,
    'norvbin': 0,
    'interlopercount': 0,
    'using': 0
}

orbit_rss = [[list() for n in range(Nsatbins)] for N in range(Nclusterbins)]
orbit_vss = [[list() for n in range(Nsatbins)] for N in range(Nclusterbins)]
orbit_ass = [[list() for n in range(Nsatbins)] for N in range(Nclusterbins)]
interloper_rss = [[np.array([]) for n in range(Nsatbins)] for N in range(Nclusterbins)]
interloper_vss = [[np.array([]) for n in range(Nsatbins)] for N in range(Nclusterbins)]

with h5py.File(cfg.outfile, 'r') as f:

    sfs = f['config/scales']
    abins = np.concatenate((
        np.array([sfs[0] - .5 * (sfs[1] - sfs[0])]),
        sfs[:-1] + 0.5 * np.diff(sfs),
        np.array([sfs[-1] + .5 * (sfs[-1] - sfs[-2])])
    ))

    print 'PROCESSING CLUSTER ORBITS', timer()

    for progress, (cluster_id, cluster) in enumerate(f['clusters'].items()):

        print '  processing orbits for cluster', progress + 1, '/', len(f['clusters']), timer()

        try:
            no_interlopers = False
            print '    ', len(cluster['satellites']), len(cluster['interlopers/ids'])
        except KeyError:
            no_interlopers = True
            print '0!'

        if np.logical_or(cluster['mvir'][-1] < pdf_cfg.pdf_m_min_cluster[0], cluster['mvir'][-1] > pdf_cfg.pdf_m_max_cluster[-1]):
            statistics['clustermass'] += len(cluster['satellites'])
            continue
        else:
            clustermass_bin = np.argmax(np.logical_and(cluster['mvir'][-1] > pdf_cfg.pdf_m_min_cluster, cluster['mvir'][-1] < pdf_cfg.pdf_m_max_cluster))

        for sat_id, sat in cluster['satellites'].items():

            if np.logical_or(sat['mvir'][-1] < pdf_cfg.pdf_m_min_satellite[0], sat['mvir'][-1] > pdf_cfg.pdf_m_max_satellite[-1]):
                statistics['satmass'] += 1
                continue
            else:
                satmass_bin = np.argmax(np.logical_and(sat['mvir'][-1] > pdf_cfg.pdf_m_min_satellite, sat['mvir'][-1] < pdf_cfg.pdf_m_max_satellite))
            
            i_infall = np.argmax(
                np.logical_and(
                    np.array(sat['sp_is_fpp']),
                    np.array(sat['superparent/ids']) > 0
                )
            )

            if i_infall >= 3:
                if np.isnan(sat['mvir'][i_infall - 3]):
                    statistics['preaccretion'] += 1
                    continue

            if sat['mvir'][i_infall] < pdf_cfg.resolution_cut:
                statistics['accretionmass'] += 1
                continue

            r, v = delta_RV(sat, cluster)
            a = sfs[i_infall]

            if (r > rbins[-1]) or (v > vbins[-1]):
                statistics['norvbin'] += 1
                continue
            
            orbit_rss[clustermass_bin][satmass_bin].append(r)
            orbit_vss[clustermass_bin][satmass_bin].append(v)
            orbit_ass[clustermass_bin][satmass_bin].append(a)
            
            statistics['using'] += 1

        if not no_interlopers:
            for satmass_bin in range(Nsatbins):
                select_interlopers = np.logical_and(
                    np.array(cluster['interlopers/mvir']) > pdf_cfg.resolution_cut,
                    np.logical_and(
                        np.array(cluster['interlopers/mvir']) > pdf_cfg.pdf_m_min_satellite[satmass_bin],
                        np.array(cluster['interlopers/mvir']) < pdf_cfg.pdf_m_max_satellite[satmass_bin]
                    )
                )

                statistics['interlopercount'] += np.sum(select_interlopers)

                more_interloper_rs, more_interloper_vs = delta_RV_interlopers(cluster)
                #more_interloper_rs = np.sqrt(np.sum(np.power(cluster['interlopers/xyz'][:, :2], 2), axis=1))[select_interlopers] / (1.E-3 * cluster['rvir'][-1])
                #more_interloper_vs = np.abs(cluster['interlopers/vxyz'][:, 2])[select_interlopers] / cluster['vrms'][-1]
                interloper_rss[clustermass_bin][satmass_bin] = np.concatenate((interloper_rss[clustermass_bin][satmass_bin], more_interloper_rs))
                interloper_vss[clustermass_bin][satmass_bin] = np.concatenate((interloper_vss[clustermass_bin][satmass_bin], more_interloper_vs))

orbit_pdfs = [[list() for n in range(Nsatbins)] for N in range(Nclusterbins)]
interloper_pdfs = [[list() for n in range(Nsatbins)] for N in range(Nclusterbins)]
for i, j in product(range(Nclusterbins), range(Nsatbins)):
    hist_input = np.vstack((orbit_vss[i][j], orbit_rss[i][j], orbit_ass[i][j])).T
    orbit_pdf, edges = np.histogramdd(hist_input, bins=(vbins, rbins, abins))
    orbit_pdfs[i][j] = orbit_pdf

    hist_input = np.vstack((interloper_vss[i][j], interloper_rss[i][j])).T
    interloper_pdf, edges = np.histogramdd(hist_input, bins=(vbins, rbins))
    interloper_pdfs[i][j] = interloper_pdf

#pdfs = np.concatenate((orbit_pdfs, interloper_pdfs[..., np.newaxis]), axis=2)

print statistics

with h5py.File(pdf_cfg.pdfsfile, 'w') as f:
    f['vbins'] = vbins
    f['rbins'] = rbins
    #f['abins'] = np.concatenate((abins, np.array([np.nan])))
    f['abins'] = np.array(abins)
    for i, j in product(range(Nclusterbins), range(Nsatbins)):
        f['orbit_pdf_{0:0d}_{1:0d}'.format(i, j)] = orbit_pdfs[i][j]
        f['interloper_pdf_{0:0d}_{1:0d}'.format(i, j)] = interloper_pdfs[i][j]

    g = f.create_group('config')
    for key, value in cfg.items():
        g.attrs.create(key, value)
    for key, value in pdf_cfg.items():
        g.attrs.create(key, value)

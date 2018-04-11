import h5py
import numpy as np
from itertools import product
from ._util import _log
from abc import ABCMeta, abstractmethod
from ._orbits import OrbitConfig

class OrbitPDFConfig(OrbitConfig):
    
    reqkeys = {
        'orbitfile',
        'pdf_m_min_satellite',
        'pdf_m_max_satellite',
        'pdf_m_min_cluster',
        'pdf_m_max_cluster',
        'resolution_cut'
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def __getattrs__(self, key):
        return self[key]
    
    def __setattr__(self, key, value):
        self[key] = value
        return

    def _validate(self):
        if set(self.keys()) < self.reqkeys:
            raise AttributeError("Missing OrbitPDFConfig attributes (check OrbitPDFConfig.reqkeys).")
        try:
            self['pdf_m_min_satellite'] = self['pdf_m_min_satellite'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for pdf_m_min_satellite (astropy.units).")
        try:
            self['pdf_m_max_satellite'] = self['pdf_m_max_satellite'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for pdf_m_max_satellite (astropy.units).")
        try:
            self['pdf_m_min_cluster'] = self['pdf_m_min_cluster'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for pdf_m_min_cluster (astropy.units).")
        try:
            self['pdf_m_max_cluster'] = self['pdf_m_max_cluster'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for pdf_m_max_cluster (astropy.units).")
        try:
            self['resolution_cut'] = self['resolution_cut'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for resolution_cut (astropy.units).")

        return
            

class _BaseOrbitPDF(object):

    __metaclass__ = ABCMeta

    def wrapbox(self, xyz):
        xyz[xyz < -self.cfg.lbox / 2.] += self.cfg.lbox
        xyz[xyz > self.cfg.lbox / 2.] -= self.cfg.lbox
        return xyz

    def delta_RV(self, sat, cluster):
        rel_xyz = wrapbox(sat['xyz'][-1] - cluster['xyz'][-1])
        return (
            np.sqrt(np.sum(np.power(rel_xyz[:2], 2))) / (1.E-3 * cluster['rvir'][-1]), 
            np.abs(sat['vxyz'][-1, 2] - cluster['vxyz'][-1, 2] + 100.0 * self.cfg.h0 * rel_xyz[2]) / cluster['vrms'][-1]
        )

    def delta_RV_interlopers(self, cluster):
        rel_xyz = wrapbox(cluster['interlopers/xyz'] - cluster['xyz'][-1])
        return (
            np.sqrt(np.sum(np.power(rel_xyz[:, :2], 2), axis=1)) / (1.E-3 * cluster['rvir'][-1]), 
            np.abs(cluster['interlopers/vxyz'][:, 2] - cluster['vxyz'][-1, 2] + 100.0 * self.cfg.h0 * rel_xyz[:, 2]) / cluster['vrms'][-1]
        )

    def __init__(self, cfg=None):

        cfg._validate()
        self.cfg = cfg
        self.Nsatbins = len(self.cfg.pdf_m_min_satellite)
        self.Nclusterbins = len(self.cfg.pdf_m_min_cluster)

        with h5py.File(self.cfg.orbitfile, 'r') as f:
            self.sfs = f['config/scales']
            self.abins = np.concatenate((
                np.array([self.sfs[0] - .5 * (self.sfs[1] - self.sfs[0])]),
                self.sfs[:-1] + 0.5 * np.diff(self.sfs),
                np.array([self.sfs[-1] + .5 * (self.sfs[-1] - self.sfs[-2])])
            ))

        self.orbit_rss = [[list() for n in range(self.Nsatbins)] for N in range(self.Nclusterbins)]
        self.orbit_vss = [[list() for n in range(self.Nsatbins)] for N in range(Nclusterbins)]
        self.orbit_Qss = [[list() for n in range(self.Nsatbins)] for N in range(Nclusterbins)]
        self.interloper_rss = [[np.array([]) for n in range(self.Nsatbins)] for N in range(self.Nclusterbins)]
        self.interloper_vss = [[np.array([]) for n in range(self.Nsatbins)] for N in range(self.Nclusterbins)]
        self.orbit_pdfs = [[list() for n in range(self.Nsatbins)] for N in range(self.Nclusterbins)]
        self.interloper_pdfs = [[list() for n in range(self.Nsatbins)] for N in range(self.Nclusterbins)]

        return

    @abstractmethod
    def calculate_q(self, sat, cluster):
        pass

    def process_clusters(self):

        _log('PROCESSING CLUSTER ORBITS')

        with h5py.File(self.cfg.orbitfile, 'r') as f:
            for progress, (cluster_id, cluster) in enumerate(f['clusters'].items()):

                _log('  processing orbits for cluster', progress + 1, '/', len(f['clusters']))

                try:
                    no_interlopers = False
                    _log('    ', len(cluster['satellites']), len(cluster['interlopers/ids']))
                except KeyError:
                    no_interlopers = True

                if np.logical_or(cluster['mvir'][-1] < self.cfg.pdf_m_min_cluster[0], \
                                 cluster['mvir'][-1] > self.cfg.pdf_m_max_cluster[-1]):
                    statistics['clustermass'] += len(cluster['satellites'])
                    continue
                else:
                    clustermass_bin = np.argmax(
                        np.logical_and(cluster['mvir'][-1] > self.cfg.pdf_m_min_cluster, \
                                       cluster['mvir'][-1] < self.cfg.pdf_m_max_cluster)
                    )

                for sat_id, sat in cluster['satellites'].items():

                    if np.logical_or(sat['mvir'][-1] < self.cfg.pdf_m_min_satellite[0], \
                                     sat['mvir'][-1] > self.cfg.pdf_m_max_satellite[-1]):
                        statistics['satmass'] += 1
                        continue
                    else:
                        satmass_bin = np.argmax(
                            np.logical_and(sat['mvir'][-1] > self.cfg.pdf_m_min_satellite, \
                                           sat['mvir'][-1] < self.cfg.pdf_m_max_satellite)
                        )

                    i_infall = np.argmax(
                        np.logical_and(np.array(sat['sp_is_fpp']), \
                                       np.array(sat['superparent/ids']) > 0)
                    )

                    if i_infall >= 3:
                        if np.isnan(sat['mvir'][i_infall - 3]):
                            statistics['preaccretion'] += 1
                            continue

                    if sat['mvir'][i_infall] < self.cfg.resolution_cut:
                        statistics['accretionmass'] += 1
                        continue

                    r, v = self.delta_RV(sat, cluster)
                    
                    q = self.compute_q(sat, cluster)

                    if (r > rbins[-1]) or (v > vbins[-1]):
                        statistics['norvbin'] += 1
                        continue

                    self.orbit_rss[clustermass_bin][satmass_bin].append(r)
                    self.orbit_vss[clustermass_bin][satmass_bin].append(v)
                    self.orbit_Qss[clustermass_bin][satmass_bin].append(a)

                    statistics['using'] += 1

                if not no_interlopers:
                    for satmass_bin in range(self.Nsatbins):
                        select_interlopers = np.logical_and(
                            np.array(cluster['interlopers/mvir']) > self.cfg.resolution_cut,
                            np.logical_and(
                                np.array(cluster['interlopers/mvir']) > self.cfg.pdf_m_min_satellite[satmass_bin],
                                np.array(cluster['interlopers/mvir']) < self.cfg.pdf_m_max_satellite[satmass_bin]
                            )
                        )

                        statistics['interlopercount'] += np.sum(select_interlopers)

                        more_interloper_rs, more_interloper_vs = self.delta_RV_interlopers(cluster)
                        self.interloper_rss[clustermass_bin][satmass_bin] = np.concatenate((interloper_rss[clustermass_bin][satmass_bin], more_interloper_rs))
                        self.interloper_vss[clustermass_bin][satmass_bin] = np.concatenate((interloper_vss[clustermass_bin][satmass_bin], more_interloper_vs))

        return

    def calculate_pdfs(self):

        for i, j in product(range(self.Nclusterbins), range(self.Nsatbins)):
            hist_input = np.vstack((
                self.orbit_vss[i][j],
                self.orbit_rss[i][j],
                self.orbit_Qss[i][j]
            )).T
            orbit_pdf, edges = np.histogramdd(hist_input, bins=(vbins, rbins, abins))
            self.orbit_pdfs[i][j] = orbit_pdf

            hist_input = np.vstack((
                self.interloper_vss[i][j], 
                self.interloper_rss[i][j]
            )).T
            interloper_pdf, edges = np.histogramdd(hist_input, bins=(vbins, rbins))
            self.interloper_pdfs[i][j] = interloper_pdf

    def write(self, outfile='pdf_out.hdf5'):
        for k, v in self.statistics:
            print(k, v)

        with h5py.File(outfile, 'w') as f:
            f['vbins'] = self.vbins
            f['rbins'] = self.rbins
            f['abins'] = np.array(self.abins)
            for i, j in product(range(self.Nclusterbins), range(self.Nsatbins)):
                f['orbit_pdf_{0:0d}_{1:0d}'.format(i, j)] = self.orbit_pdfs[i][j]
                f['interloper_pdf_{0:0d}_{1:0d}'.format(i, j)] = self.interloper_pdfs[i][j]

            g = f.create_group('config')
            for key, value in self.cfg.items():
                g.attrs.create(key, value)

        _log('END')

        return

class InfallTimeOrbitPDF(_BaseOrbitPDF):

    def __init__(self, cfg=None):
        
        super().__init__(cfg=cfg)

        return


    def calculate_q(self, sat, cluster):
        
        i_infall = np.argmax(
            np.logical_and(np.array(sat['sp_is_fpp']), \
                           np.array(sat['superparent/ids']) > 0)
        )
        
        return self.sfs[i_infall]
        
        

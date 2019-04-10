import h5py
import numpy as np
from itertools import product
from ._util import _log
from abc import ABCMeta, abstractmethod

np.seterr(all='ignore')


class _BaseOrbitPDF(object):

    __metaclass__ = ABCMeta

    def wrapbox(self, xyz):
        xyz[xyz < -self.cfg.lbox.value / 2.] += self.cfg.lbox.value
        xyz[xyz > self.cfg.lbox.value / 2.] -= self.cfg.lbox.value
        return xyz

    def delta_RV(self, sat, cluster):
        rel_xyz = self.wrapbox(sat['xyz'][-1] - cluster['xyz'][-1])
        return (
            np.sqrt(np.sum(np.power(rel_xyz[:2], 2))) /
            (1.E-3 * cluster['rvir'][-1]),
            np.abs(sat['vxyz'][-1, 2] - cluster['vxyz'][-1, 2] +
                   100.0 * self.cfg.h0 * rel_xyz[2]) / cluster['vrms'][-1]
        )

    def delta_RV_interlopers(self, cluster):
        rel_xyz = self.wrapbox(cluster['interlopers/xyz'] - cluster['xyz'][-1])
        return (
            np.sqrt(np.sum(np.power(rel_xyz[:, :2], 2), axis=1)) /
            (1.E-3 * cluster['rvir'][-1]),
            np.abs(cluster['interlopers/vxyz'][:, 2] - cluster['vxyz'][-1, 2] +
                   100.0 * self.cfg.h0 * rel_xyz[:, 2]) / cluster['vrms'][-1]
        )

    def __init__(self, cfg=None):

        cfg._validate()
        self.cfg = cfg
        self.Nsatbins = len(self.cfg.pdf_m_min_satellite)
        self.Nclusterbins = len(self.cfg.pdf_m_min_cluster)

        self.init_qbins()

        self.orbit_rss = [
            [list() for n in range(self.Nsatbins)]
            for N in range(self.Nclusterbins)
        ]
        self.orbit_vss = [
            [list() for n in range(self.Nsatbins)]
            for N in range(self.Nclusterbins)
        ]
        self.orbit_qss = [
            [list() for n in range(self.Nsatbins)]
            for N in range(self.Nclusterbins)
        ]
        self.interloper_rss = [
            [np.array([]) for n in range(self.Nsatbins)]
            for N in range(self.Nclusterbins)
        ]
        self.interloper_vss = [
            [np.array([]) for n in range(self.Nsatbins)]
            for N in range(self.Nclusterbins)
        ]
        self.orbit_pdfs = [
            [list() for n in range(self.Nsatbins)]
            for N in range(self.Nclusterbins)
        ]
        self.interloper_pdfs = [
            [list() for n in range(self.Nsatbins)]
            for N in range(self.Nclusterbins)
        ]

        self.statistics = {
            'clustermass': 0,
            'satmass': 0,
            'preaccretion': 0,
            'accretionmass': 0,
            'norvbin': 0,
            'interlopercount': 0,
            'qisnan': 0,
            'using': 0
        }

        return

    @abstractmethod
    def calculate_q(self, sat, cluster):
        pass

    @abstractmethod
    def init_qbins(self):
        pass

    def process_orbits(self):

        _log('PROCESSING CLUSTER ORBITS')

        with h5py.File(self.cfg.orbitfile, 'r') as f:
            for progress, (cluster_id, cluster) in enumerate(
                    f['clusters'].items()):

                _log(
                    '  processing orbits for cluster',
                    progress + 1,
                    '/',
                    len(f['clusters'])
                )

                try:
                    no_interlopers = False
                    _log(
                        '    Nsat',
                        len(cluster['satellites']),
                        'Ninterloper',
                        len(cluster['interlopers/ids'])
                    )
                except KeyError:
                    no_interlopers = True

                if np.logical_or(
                        cluster['mvir'][-1]
                        < self.cfg.pdf_m_min_cluster[0].value,
                        cluster['mvir'][-1]
                        > self.cfg.pdf_m_max_cluster[-1].value
                ):
                    self.statistics['clustermass'] += \
                        len(cluster['satellites'])
                    continue
                else:
                    clustermass_bin = np.argmax(
                        np.logical_and(
                            cluster['mvir'][-1]
                            > self.cfg.pdf_m_min_cluster.value,
                            cluster['mvir'][-1]
                            < self.cfg.pdf_m_max_cluster.value
                        )
                    )

                for sat_id, sat in cluster['satellites'].items():

                    if np.logical_or(
                            sat['mvir'][-1]
                            < self.cfg.pdf_m_min_satellite[0].value,
                            sat['mvir'][-1]
                            > self.cfg.pdf_m_max_satellite[-1].value
                    ):
                        self.statistics['satmass'] += 1
                        continue
                    else:
                        satmass_bin = np.argmax(
                            np.logical_and(
                                sat['mvir'][-1]
                                > self.cfg.pdf_m_min_satellite.value,
                                sat['mvir'][-1]
                                < self.cfg.pdf_m_max_satellite.value
                            )
                        )

                    i_infall = np.argmax(
                        np.logical_and(np.array(sat['sp_is_fpp']),
                                       np.array(sat['superparent/ids']) > 0)
                    )

                    if i_infall >= 3:
                        if np.isnan(sat['mvir'][i_infall - 3]):
                            self.statistics['preaccretion'] += 1
                            continue

                    if sat['mvir'][i_infall] < self.cfg.resolution_cut.value:
                        self.statistics['accretionmass'] += 1
                        continue

                    r, v = self.delta_RV(sat, cluster)
                    q = self.calculate_q(sat, cluster)

                    if (r > self.cfg.rbins[-1]) or (v > self.cfg.vbins[-1]):
                        self.statistics['norvbin'] += 1
                        continue

                    if np.isnan(q):
                        self.statistics['qisnan'] += 1
                        continue

                    self.orbit_rss[clustermass_bin][satmass_bin].append(r)
                    self.orbit_vss[clustermass_bin][satmass_bin].append(v)
                    self.orbit_qss[clustermass_bin][satmass_bin].append(q)

                    self.statistics['using'] += 1

                if not no_interlopers:
                    for satmass_bin in range(self.Nsatbins):
                        select_interlopers = np.logical_and(
                            np.array(cluster['interlopers/mvir'])
                            > self.cfg.resolution_cut.value,
                            np.logical_and(
                                np.array(cluster['interlopers/mvir'])
                                > self.cfg.pdf_m_min_satellite[satmass_bin]
                                .value,
                                np.array(cluster['interlopers/mvir'])
                                < self.cfg.pdf_m_max_satellite[satmass_bin]
                                .value
                            )
                        )

                        self.statistics['interlopercount'] += \
                            np.sum(select_interlopers)

                        more_interloper_rs, more_interloper_vs = \
                            self.delta_RV_interlopers(cluster)
                        self.interloper_rss[clustermass_bin][satmass_bin] = \
                            np.concatenate((
                                self.interloper_rss[clustermass_bin][
                                    satmass_bin],
                                more_interloper_rs[select_interlopers]
                            ))
                        self.interloper_vss[clustermass_bin][satmass_bin] = \
                            np.concatenate((
                                self.interloper_vss[clustermass_bin][
                                    satmass_bin],
                                more_interloper_vs[select_interlopers]
                            ))

        return

    def calculate_pdfs(self):

        for i, j in product(range(self.Nclusterbins), range(self.Nsatbins)):
            hist_input = np.vstack((
                self.orbit_vss[i][j],
                self.orbit_rss[i][j],
                self.orbit_qss[i][j]
            )).T
            orbit_pdf, edges = np.histogramdd(
                hist_input,
                bins=(self.cfg.vbins, self.cfg.rbins, self.qbins)
            )
            self.orbit_pdfs[i][j] = orbit_pdf

            hist_input = np.vstack((
                self.interloper_vss[i][j],
                self.interloper_rss[i][j]
            )).T
            interloper_pdf, edges = np.histogramdd(
                hist_input,
                bins=(self.cfg.vbins, self.cfg.rbins)
            )
            self.interloper_pdfs[i][j] = interloper_pdf

    def write(self):
        for k, v in self.statistics.items():
            print(k, v)

        with h5py.File(self.cfg.pdfsfile, 'w') as f:
            f['vbins'] = self.cfg.vbins
            f['rbins'] = self.cfg.rbins
            f['qbins'] = np.array(self.qbins)
            for i, j in product(
                    range(self.Nclusterbins), range(self.Nsatbins)):
                f['orbit_pdf_{0:0d}_{1:0d}'.format(i, j)] = \
                    self.orbit_pdfs[i][j]
                f['interloper_pdf_{0:0d}_{1:0d}'.format(i, j)] = \
                    self.interloper_pdfs[i][j]

            g = f.create_group('config')
            for key, value in self.cfg.items():
                g.attrs[key] = value

        _log('END')

        return


class InfallTimeOrbitPDF(_BaseOrbitPDF):

    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        return

    def init_qbins(self):
        with h5py.File(self.cfg.orbitfile, 'r') as f:
            self.sfs = np.array(f['config/scales'])
            self.qbins = np.concatenate((
                np.array([self.sfs[0] - .5 * (self.sfs[1] - self.sfs[0])]),
                self.sfs[:-1] + 0.5 * np.diff(self.sfs),
                np.array([self.sfs[-1] + .5 * (self.sfs[-1] - self.sfs[-2])])
            ))
            return

    def calculate_q(self, sat, cluster):
        i_infall = np.argmax(
            np.logical_and(np.array(sat['sp_is_fpp']),
                           np.array(sat['superparent/ids']) > 0)
        )
        return self.sfs[i_infall]


class RperiOrbitPDF(_BaseOrbitPDF):

    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        return

    def init_qbins(self):
        self.qbins = np.linspace(0, 2.5, 51)
        return

    def calculate_q(self, sat, cluster):
        rel_xyz = self.wrapbox(
            np.array(sat['xyz'])
            - np.array(cluster['xyz'])
        ) / (1.E-3 * cluster['rvir'][-1])
        return np.sqrt(np.nanmin(np.sum(np.power(rel_xyz, 2), axis=1)))


class PeriTimeOrbitPDF(_BaseOrbitPDF):

    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        return

    def init_qbins(self):
        with h5py.File(self.cfg.orbitfile, 'r') as f:
            self.sfs = np.array(f['config/scales'])
            self.qbins = np.concatenate((
                np.array([self.sfs[0] - .5 * (self.sfs[1] - self.sfs[0])]),
                self.sfs[:-1] + 0.5 * np.diff(self.sfs),
                np.array([self.sfs[-1] + .5 * (self.sfs[-1] - self.sfs[-2])])
            ))
            return

    def calculate_q(self, sat, cluster):
        rel_xyz = self.wrapbox(
            np.array(sat['xyz'])
            - np.array(cluster['xyz'])
        ) / (1.E-3 * cluster['rvir'][-1])
        rel_r = np.sqrt(np.sum(np.power(rel_xyz, 2), axis=1))
        minima = np.logical_and(
            np.r_[1, rel_r[1:] < rel_r[:-1]],
            np.r_[rel_r[:-1] < rel_r[1:], 1]
        )
        minima[-1] = False
        rcut = np.max(self.cfg.rbins)
        minima[rel_r > rcut] = False
        if np.sum(minima) > 0:
            return np.min(self.sfs[minima])
        else: 
            # no pericentre, e.g. circular orbit or 
            # insufficient future time, in initial test
            # this seems to be a small minority for
            # reasonable choices
            return np.nan
        

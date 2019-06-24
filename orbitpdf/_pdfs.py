import h5py
import numpy as np
from ._util import _log
import astropy.units as U

np.seterr(all='ignore')


class OrbitPDF(object):

    def wrapbox(self, xyz):
        xyz[xyz < -self.cfg.lbox.value / 2.] += self.cfg.lbox.value
        xyz[xyz > self.cfg.lbox.value / 2.] -= self.cfg.lbox.value
        return xyz

    def delta_RV(self, sat, cluster):
        rel_xyz = self.wrapbox(sat['xyz'][self.iref]
                               - cluster['xyz'][self.iref])
        return (
            np.sqrt(np.sum(np.power(rel_xyz[:2], 2))) /
            (1.E-3 * cluster['rvir'][self.iref]),
            np.abs(sat['vxyz'][self.iref, 2] - cluster['vxyz'][self.iref, 2] +
                   100.0 * self.cfg.h0 * rel_xyz[2])
            / cluster['vrms'][self.iref]
        )

    def delta_RV_interlopers(self, cluster):
        rel_xyz = self.wrapbox(cluster['interlopers/xyz'] -
                               cluster['xyz'][self.iref])
        return (
            np.sqrt(np.sum(np.power(rel_xyz[:, :2], 2), axis=1)) /
            (1.E-3 * cluster['rvir'][self.iref]),
            np.abs(cluster['interlopers/vxyz'][:, 2]
                   - cluster['vxyz'][self.iref, 2] +
                   100.0 * self.cfg.h0 * rel_xyz[:, 2])
            / cluster['vrms'][self.iref]
        )

    def __init__(self, cfg=None):

        cfg._validate()
        self.cfg = cfg
        self.iref = -1 - self.cfg.skipmore_for_select
        with h5py.File(self.cfg.orbitfile, 'r') as f:
            self.sfs = np.array(f['config/scales'])

        self.no_interlopers = None  # checked when processing orbits

        self.rlist = []
        self.vlist = []
        self.mhostlist = []
        self.msatlist = []
        self.qlists = {k: [] for k in self.qkeys}

        self.rlist_i = np.array([])
        self.vlist_i = np.array([])
        self.mhostlist_i = np.array([])
        self.msatlist_i = np.array([])

        self.statistics = {
            'clustermass': 0,
            'satmass': 0,
            'preaccretion': 0,
            'peakmass': 0,
            'norvbin': 0,
            'interlopercount': 0,
            'using': 0
        }

        return

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
                    self.no_interlopers = False
                    _log(
                        '    Nsat',
                        len(cluster['satellites']),
                        'Ninterloper',
                        len(cluster['interlopers/ids'])
                    )
                except KeyError:
                    self.no_interlopers = True

                if np.logical_or(
                        cluster['mvir'][self.iref]
                        < self.cfg.pdf_m_min_cluster.value,
                        cluster['mvir'][self.iref]
                        > self.cfg.pdf_m_max_cluster.value
                ):
                    self.statistics['clustermass'] += \
                        len(cluster['satellites'])
                    continue

                for sat_id, sat in cluster['satellites'].items():

                    if np.logical_or(
                            sat['mvir'][self.iref]
                            < self.cfg.pdf_m_min_satellite.value,
                            sat['mvir'][self.iref]
                            > self.cfg.pdf_m_max_satellite.value
                    ):
                        self.statistics['satmass'] += 1
                        continue

                    i_infall = np.argmax(np.logical_and(
                        np.array(sat['sp_is_fpp'][:self.iref]),
                        np.array(sat['superparent/ids'][self.iref]) > 0
                    ))

                    if i_infall >= 3:
                        if np.isnan(sat['mvir'][i_infall - 3]):
                            self.statistics['preaccretion'] += 1
                            continue

                    if np.max(sat['mvir']) < self.cfg.resolution_cut.value:
                        self.statistics['peakmass'] += 1
                        continue

                    r, v = self.delta_RV(sat, cluster)

                    if (r > self.cfg.interloper_dR) or \
                       (v > self.cfg.interloper_dV):
                        self.statistics['norvbin'] += 1
                        continue

                    self.rlist.append(r)
                    self.vlist.append(v)
                    self.mhostlist.append(cluster['mvir'][self.iref])
                    self.msatlist.append(sat['mvir'][self.iref])
                    qs = self.calculate_q(sat, cluster)
                    for k in self.qkeys:
                        self.qlists[k].append(qs[k])

                    self.statistics['using'] += 1

                if not self.no_interlopers:
                    select_interlopers = np.logical_and(
                        np.array(cluster['interlopers/mvir'])
                        > self.cfg.resolution_cut.value,
                        np.logical_and(
                            np.array(cluster['interlopers/mvir'])
                            > self.cfg.pdf_m_min_satellite
                            .value,
                            np.array(cluster['interlopers/mvir'])
                            < self.cfg.pdf_m_max_satellite
                            .value
                        )
                    )

                    self.statistics['interlopercount'] += \
                        np.sum(select_interlopers)

                    more_interloper_rs, more_interloper_vs = \
                        self.delta_RV_interlopers(cluster)
                    self.rlist_i = np.concatenate(
                        (self.rlist_i, more_interloper_rs[select_interlopers])
                    )
                    self.vlist_i = np.concatenate(
                        (self.vlist_i, more_interloper_vs[select_interlopers])
                    )
                    self.mhostlist_i = np.concatenate((
                        self.mhostlist_i,
                        np.ones(np.sum(select_interlopers))
                        * cluster['mvir'][self.iref]
                    ))
                    self.msatlist_i = np.concatenate((
                        self.msatlist_i,
                        cluster['interlopers/mvir'][select_interlopers]
                    ))
        return

    def write(self):
        for k, v in self.statistics.items():
            print(k, v)

        with h5py.File(self.cfg.pdfsfile, 'w') as f:
            g = f.create_group('satellites')
            g['R'] = np.array(self.rlist)
            g['R'].attrs['unit'] = str(U.dimensionless_unscaled)
            g['R'].attrs['desc'] = 'Projected offset from host, R/Rvir,' \
                ' with Rvir defined as in Bryan & Norman (1998).'
            g['V'] = np.array(self.vlist)
            g['V'].attrs['unit'] = str(U.dimensionless_unscaled)
            g['V'].attrs['desc'] = 'Line-of-sight velocity offset from host,' \
                ' V/sigma, with sigma the *3D* cluster velocity dispersion.'
            g['Mhost'] = np.array(self.mhostlist)
            g['Mhost'].attrs['unit'] = str(U.Msun)
            g['Mhost'].attrs['desc'] = 'Host halo virial mass, with virial' \
                ' as defined in Bryan & Norman (1998).'
            g['Msat'] = np.array(self.msatlist)
            g['Msat'].attrs['unit'] = str(U.Msun)
            g['Msat'].attrs['desc'] = 'Satellite halo virial mass, with' \
                ' virial as defined in Bryan & Norman (1998).'
            for k in self.qkeys:
                g[k] = np.array(self.qlists[k])
                g[k].attrs['unit'] = str(self.qunits[k])
                g[k].attrs['desc'] = self.qdescs[k]
            if not self.no_interlopers:
                g = f.create_group('interlopers')
                g['R'] = np.array(self.rlist_i)
                g['R'].attrs['unit'] = str(U.dimensionless_unscaled)
                g['R'].attrs['desc'] = 'Projected offset from host, R/Rvir,' \
                    ' with Rvir defined as in Bryan & Norman (1998).'
                g['V'] = np.array(self.vlist_i)
                g['V'].attrs['unit'] = str(U.dimensionless_unscaled)
                g['V'].attrs['desc'] = 'Line-of-sight velocity offset from' \
                    ' host, V/sigma, with sigma the *3D* cluster velocity' \
                    ' dispersion.'
                g['Mhost'] = np.array(self.mhostlist_i)
                g['Mhost'].attrs['unit'] = str(U.Msun)
                g['Mhost'].attrs['unit'] = 'Host halo virial mass, with' \
                    ' virial as defined in Bryan & Norman (1998).'
                g['Msat'] = np.array(self.msatlist_i)
                g['Msat'].attrs['unit'] = str(U.Msun)
                g['Msat'].attrs['unit'] = 'Interloper halo virial mass, with' \
                    ' virial as defined in Bryan & Norman (1998).'
            g = f.create_group('config')
            for key, value in self.cfg.items():
                g.attrs[key] = value

        _log('END')

        return

    qkeys = {'t_infall', 't_peri', 'r', 'v', 'r_min', 'v_max'}
    qunits = {
        't_infall': 'dimensionless_unscaled',
        't_peri': 'dimensionless_unscaled',
        'r': 'dimensionless_unscaled',
        'v': 'dimensionless_unscaled',
        'r_min': 'dimensionless_unscaled',
        'v_max': 'dimensionless_unscaled'
    }
    qdescs = {
        't_infall': 'Scale factor at first infall into final host'
        ' defined as crossing cfg.interloper_dR times the virial'
        ' radius, where the radius is fixed to its value at the'
        ' reference time, and virial is defined as in Bryan & Norman (1998).',
        't_peri': 'Scale factor at first pericentre within final host.',
        'r': 'Current deprojected radius, scaled to host virial radius,'
        ' where virial is defined as in Bryan & Norman (1998).',
        'v': 'Current deprojected velocity, scaled to host *3D* velocity'
        ' dispersion.',
        'r_min': 'Distance of closest approach to final host, scaled to the'
        ' host virial radius, where virial is defined as in Bryan & Norman'
        ' (1998).',
        'v_max': 'Maximum velocity relative to final host, scaled to the host'
        ' *3D* velocity dispersion.'
    }

    def calculate_q(self, sat, cluster):
        retval = dict()

        rel_xyz = self.wrapbox(
            np.array(sat['xyz'])
            - np.array(cluster['xyz'])
        ) / (1.E-3 * cluster['rvir'][self.iref])
        rel_r = np.sqrt(np.sum(np.power(rel_xyz, 2), axis=1))
        rel_vxyz = (
            np.array(sat['vxyz'])
            - np.array(cluster['vxyz'])
        ) / cluster['vrms'][self.iref]
        rel_v = np.sqrt(np.sum(np.power(rel_vxyz, 2), axis=1))

        # INFALL TIME
        # infall defined as crossing dR * Rvir
        # (non-evolving Rvir to avoid cluster growing and gaining sats
        # in strange places; recenter at merger might still be a bit odd)
        try:
            i_infall = np.argwhere(np.logical_and(
                rel_r[:-1] > self.cfg.interloper_dR,
                rel_r[1:] < self.cfg.interloper_dR
            )).flatten()[0]
        except IndexError:
            retval['t_infall'] = np.nan
        else:
            mask = np.s_[i_infall: i_infall + 1]
            retval['t_infall'] = \
                np.interp(self.cfg.interloper_dR, rel_r[mask], self.sfs[mask])

        # CLOSEST APPROACH AND MAX SPEED SO FAR
        # closest recorded approach and speed up to present time
        mask = np.s_[:] if self.iref == -1 else np.s_[:self.iref + 1]
        retval['r_min'] = np.nanmin(rel_r[mask])
        retval['v_max'] = np.nanmax(rel_v[mask])

        # TIME OF FIRST PERICENTRE
        minima = np.logical_and(
            np.r_[1, rel_r[1:] < rel_r[:-1]],
            np.r_[rel_r[:-1] < rel_r[1:], 1]
        )
        minima[-1] = False
        minima[rel_r > self.cfg.interloper_dR] = False
        if np.sum(minima) > 0:
            retval['t_peri'] = np.min(self.sfs[minima])
        else:
            # no pericentre, e.g. circular orbit or
            # insufficient future time, in initial test
            # this seems to be a small minority for
            # reasonable choices
            retval['t_peri'] = np.nan

        # CURRENT POSITIONS
        retval['r'] = rel_r[self.iref]
        retval['v'] = rel_v[self.iref]

        return retval

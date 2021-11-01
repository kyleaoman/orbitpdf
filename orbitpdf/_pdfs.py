import h5py
import numpy as np
from ._util import _log
import astropy.units as U
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm
import sys

np.seterr(all='ignore')

qkeys = {'t_infall', 't_peri', 'r', 'v', 'r_min', 'v_max',
         'm_max', 'm_infall'}
qunits = {
    't_infall': 'dimensionless_unscaled',
    't_peri': 'dimensionless_unscaled',
    'r': 'dimensionless_unscaled',
    'v': 'dimensionless_unscaled',
    'r_min': 'dimensionless_unscaled',
    'v_max': 'dimensionless_unscaled',
    'm_max': 'solMass',
    'm_infall': 'solMass'
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
    ' *3D* velocity dispersion.',
    'm_max': 'Maximum past subhalo mass.',
    'm_infall': 'Subhalo mass at infall time.'
}


def wrapbox(xyz, lbox=None):
    xyz[xyz < -lbox.value / 2.] += lbox.value
    xyz[xyz > lbox.value / 2.] -= lbox.value
    return xyz


def delta_RV(sat, cluster, iref=None, lbox=None, H=None, signed_V=None,
             z=None):
    rel_xyz = wrapbox(sat['xyz'][iref] - cluster['xyz'][iref], lbox=lbox)
    rel_vxyz = sat['vxyz'][iref] - cluster['vxyz'][iref]
    R = np.sqrt(np.sum(np.power(rel_xyz[:2], 2))) \
        / (1.E-3 * cluster['rvir'][iref])
    a = 1 / (1 + z)
    V = np.abs(
        rel_vxyz[2] + H.to(U.km / U.s / U.Mpc).value * a * rel_xyz[2]
    ) / cluster['vrms'][iref]
    if signed_V:
        sgn = np.sign(
            rel_xyz[2] * (
                rel_vxyz[2] + rel_xyz[2] * a * H.to(U.km / U.s / U.Mpc).value
            )
        )
        V = sgn * V
    return R, V


def delta_RV_interlopers(cluster, iref=None, lbox=None, H=None,
                         signed_V=None, z=None):
    rel_xyz = wrapbox(
        cluster['interlopers/xyz'] - cluster['xyz'][iref],
        lbox=lbox
    )
    rel_vxyz = cluster['interlopers/vxyz'] - cluster['vxyz'][iref]
    R = np.sqrt(np.sum(np.power(rel_xyz[:, :2], 2), axis=1)) \
        / (1.E-3 * cluster['rvir'][iref])
    a = 1 / (1 + z)
    V = np.abs(
        rel_vxyz[:, 2] + H.to(U.km / U.s / U.Mpc).value * a * rel_xyz[:, 2]
    ) / cluster['vrms'][iref]
    if signed_V:
        sgn = np.sign(rel_xyz[:, 2] * rel_vxyz[:, 2])
        V = sgn * V
    return R, V


class OrbitPDF(object):

    def __init__(self, cfg=None):

        cfg._validate()
        self.cfg = cfg
        self.iref = -1 - self.cfg.skipmore_for_select
        with h5py.File(self.cfg.orbitfile, 'r') as f:
            self.sfs = np.array(f['config/scales'])

        self.rlist = list()
        self.vlist = list()
        self.mhostlist = list()
        self.msatlist = list()
        self.qlists = None

        self.rlist_i = list()
        self.vlist_i = list()
        self.mhostlist_i = list()
        self.msatlist_i = list()

        return

    def process_orbits(self):

        _log('PROCESSING CLUSTER ORBITS')

        with h5py.File(self.cfg.orbitfile, 'r') as f:
            clist = list(f['/clusters'].keys())

        target_kwargs = dict(self.cfg, iref=self.iref, sfs=self.sfs, )

        def target(cid):
            return _process_orbit(cid, **target_kwargs)

        if self.cfg.ncpu > 1:
            with ProcessPool(nodes=min(self.cfg.ncpu, len(clist))) as pool:
                # output = pool.imap(target, clist)  # lazy evaluation
                # output = list(output)  # force evaluation
                output = list(tqdm(
                    pool.imap(target, clist),
                    total=len(clist),
                    file=sys.stdout
                ))

        else:
            output = list()
            for progress, cid in enumerate(clist):
                if progress % 100 == 0:
                    _log(progress, '/', len(clist))
                output.append(target(cid))
        self.rlist = np.concatenate([o[0] for o in output])
        self.vlist = np.concatenate([o[1] for o in output])
        self.mhostlist = np.concatenate([o[2] for o in output])
        self.msatlist = np.concatenate([o[3] for o in output])
        self.qlists = np.concatenate([o[4] for o in output])
        self.rlist_i = np.concatenate([o[5] for o in output])
        self.vlist_i = np.concatenate([o[6] for o in output])
        self.mhostlist_i = np.concatenate([o[7] for o in output])
        self.msatlist_i = np.concatenate([o[8] for o in output])
        self.statistics = np.concatenate([o[9] for o in output])
        self.statistics = {k: self.statistics[k].sum()
                           for k in self.statistics.dtype.names}

        return

    def write(self):
        for k, v in self.statistics.items():
            _log(k, v)

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
            for k in qkeys:
                g[k] = np.array(self.qlists[k])
                g[k].attrs['unit'] = str(qunits[k])
                g[k].attrs['desc'] = qdescs[k]
            if self.statistics['interlopercount'] > 0:
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


def calculate_q(sat, cluster, iref=None, lbox=None, interloper_dR=None,
                sfs=None):
    retval = np.array(
        np.ones(1) * np.nan,
        dtype=np.dtype([(k, np.float) for k in qkeys])
    )
    rel_xyz = wrapbox(
        np.array(sat['xyz']) - np.array(cluster['xyz']),
        lbox=lbox
    ) / (1.E-3 * cluster['rvir'][iref])
    rel_r = np.sqrt(np.sum(np.power(rel_xyz, 2), axis=1))
    rel_vxyz = (
        np.array(sat['vxyz'])
        - np.array(cluster['vxyz'])
    ) / cluster['vrms'][iref]
    rel_v = np.sqrt(np.sum(np.power(rel_vxyz, 2), axis=1))

    # INFALL TIME & MASS AT THAT TIME
    # infall defined as crossing dR * Rvir
    # (non-evolving Rvir to avoid cluster growing and gaining sats
    # in strange places; recenter at merger might still be a bit odd)
    try:
        i_infall = np.argwhere(np.logical_and(
            rel_r[:-1] > interloper_dR,
            rel_r[1:] < interloper_dR
        )).flatten()[0]
    except IndexError:
        pass  # values default to nan
    else:
        mask = np.s_[i_infall: i_infall + 1]
        retval['t_infall'] = np.interp(
            interloper_dR,
            rel_r[mask],
            sfs[mask]
        )
        retval['m_infall'] = np.interp(
            retval['t_infall'],
            sfs[mask],
            sat['mvir'][mask]
        )

    # CLOSEST APPROACH AND MAX SPEED SO FAR, AND MAX PAST MASS
    # closest recorded approach and speed up to present time,
    # and maximum mass at any past time
    mask = np.s_[:] if iref == -1 else np.s_[:iref + 1]
    retval['r_min'] = np.nanmin(rel_r[mask])
    retval['v_max'] = np.nanmax(rel_v[mask])
    retval['m_max'] = np.nanmax(sat['mvir'][mask])

    # TIME OF FIRST PERICENTRE
    minima = np.logical_and(
        np.r_[1, rel_r[1:] < rel_r[:-1]],
        np.r_[rel_r[:-1] < rel_r[1:], 1]
    )
    minima[-1] = False
    minima[rel_r > interloper_dR] = False
    if np.sum(minima) > 0:
        retval['t_peri'] = np.min(sfs[minima])
    else:
        # no pericentre, e.g. circular orbit or
        # insufficient future time, in initial test
        # this seems to be a small minority for
        # reasonable choices
        retval['t_peri'] = np.nan

    # CURRENT POSITIONS
    retval['r'] = rel_r[iref]
    retval['v'] = rel_v[iref]

    return retval


def _process_orbit(cluster_id, iref=None, orbitfile=None,
                   pdf_m_min_cluster=None, pdf_m_max_cluster=None,
                   pdf_m_min_satellite=None, pdf_m_max_satellite=None,
                   resolution_cut=None, interloper_dR=None,
                   interloper_dV=None, lbox=None, H=None,
                   signed_V=None, sfs=None, z=None, **kwargs):
    statistics = np.array(
        np.zeros(1),
        dtype=np.dtype([
            ('clustermass', np.int),
            ('satmass', np.int),
            ('preaccretion', np.int),
            ('peakmass', np.int),
            ('norvbin', np.int),
            ('interlopercount', np.int),
            ('using', np.int)
        ])
    )
    rlist = list()
    vlist = list()
    mhostlist = list()
    msatlist = list()
    qlists = list()

    with h5py.File(orbitfile, 'r') as f:
        cluster = f['/clusters/{:s}'.format(cluster_id)]
        try:
            cluster['interlopers/ids']
        except KeyError:
            no_interlopers = True
        else:
            no_interlopers = False

        if np.logical_or(
                cluster['mvir'][iref]
                < pdf_m_min_cluster.value,
                cluster['mvir'][iref]
                > pdf_m_max_cluster.value
        ):
            statistics['clustermass'] += \
                len(cluster['satellites'])
            rlist = np.empty((0, ))
            vlist = np.empty((0, ))
            mhostlist = np.empty((0, ))
            msatlist = np.empty((0, ))
            qlists = np.empty(
                (0, ),
                dtype=np.dtype([(k, np.float) for k in qkeys])
            )
            rlist_i = np.empty((0, ))
            vlist_i = np.empty((0, ))
            mhostlist_i = np.empty((0, ))
            msatlist_i = np.empty((0, ))
            return rlist, vlist, mhostlist, msatlist, qlists, \
                rlist_i, vlist_i, mhostlist_i, msatlist_i, statistics

        for sat_id, sat in cluster['satellites'].items():

            if np.logical_or(
                    sat['mvir'][iref]
                    < pdf_m_min_satellite.value,
                    sat['mvir'][iref]
                    > pdf_m_max_satellite.value
            ):
                statistics['satmass'] += 1
                continue

            i_infall = np.argmax(np.logical_and(
                np.array(sat['sp_is_fpp'][:iref]),
                np.array(sat['superparent/ids'][iref]) > 0
            ))

            if i_infall >= 3:
                if np.isnan(sat['mvir'][i_infall - 3]):
                    statistics['preaccretion'] += 1
                    continue

            if np.nanmax(sat['mvir']) < resolution_cut.value:
                statistics['peakmass'] += 1
                continue

            r, v = delta_RV(sat, cluster, iref=iref, lbox=lbox, H=H,
                            signed_V=signed_V, z=z)

            if (r > interloper_dR) or \
               (np.abs(v) > interloper_dV):
                statistics['norvbin'] += 1
                continue
            rlist.append(r)
            vlist.append(v)
            mhostlist.append(cluster['mvir'][iref])
            msatlist.append(sat['mvir'][iref])
            qlists.append(calculate_q(sat, cluster, iref=iref, lbox=lbox,
                                      interloper_dR=interloper_dR, sfs=sfs))

            statistics['using'] += 1

        rlist = np.array(rlist)
        vlist = np.array(vlist)
        mhostlist = np.array(mhostlist)
        msatlist = np.array(msatlist)
        if len(qlists):
            qlists = np.concatenate(qlists)
        else:
            qlists = np.empty(
                (0, ),
                dtype=np.dtype([(k, np.float) for k in qkeys])
            )

        if not no_interlopers:
            select_interlopers = np.logical_and(
                np.array(cluster['interlopers/mvir'])
                > resolution_cut.value,
                np.logical_and(
                    np.array(cluster['interlopers/mvir'])
                    > pdf_m_min_satellite.value,
                    np.array(cluster['interlopers/mvir'])
                    < pdf_m_max_satellite.value
                )
            )

            statistics['interlopercount'] += \
                np.sum(select_interlopers)

            more_interloper_rs, more_interloper_vs = \
                delta_RV_interlopers(cluster, iref=iref, lbox=lbox, H=H,
                                     signed_V=signed_V, z=z)
            rlist_i = more_interloper_rs[select_interlopers]
            vlist_i = more_interloper_vs[select_interlopers]
            mhostlist_i = np.ones(np.sum(select_interlopers)) \
                * cluster['mvir'][iref]
            msatlist_i = cluster['interlopers/mvir'][select_interlopers]
        else:
            rlist_i = np.empty((0, ))
            vlist_i = np.empty((0, ))
            mhostlist_i = np.empty((0, ))
            msatlist_i = np.empty((0, ))

        return rlist, vlist, mhostlist, msatlist, qlists, \
            rlist_i, vlist_i, mhostlist_i, msatlist_i, statistics

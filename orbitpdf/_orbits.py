import numpy as np
import h5py
import astropy.units as U
import itertools
import os
from ._util import _log
from pathos.multiprocessing import ProcessPool

import read_tree

def _get_superparent(halo):

    if halo is None:
        return None

    if halo.parent is None:
        return None

    retval = halo.parent
    while retval.parent is not None:
        retval = retval.parent

    return retval

class Orbits(object):

    def __init__(self, cfg=None):

        _log('START')
        cfg._validate()
        self.cfg = cfg
        self.cfg.scales = np.loadtxt(self.cfg.scalefile, unpack=True, usecols=[1])
        self._find_files()
        self._write_headers()
            
        return

    def cluster_search(self):

        _log('CLUSTER SEARCH')
        pool = ProcessPool(nodes = min(self.cfg.ncpu, len(self.infiles)))
        target_kwargs = dict(self.cfg)
        target = lambda infile: _process_clusters(infile, **target_kwargs)
        all_out_arrays = pool.map(target, self.infiles)

        _log('CLUSTER REDUCTION')
        all_out_arrays = reduce(lambda a, b: a + b, all_out_arrays)

        _log('CLUSTER OUTPUT')
        with h5py.File(self.cfg.outfile, 'a') as f:
            for out_array in all_out_arrays:
                self._write_cluster(f, out_array)
                
        return

    def interloper_search(self):

        _log('INTERLOPER SEARCH')
        pool = ProcessPool(ncpus = min(self.cfg.ncpu, len(self.infiles)))
        target_kwargs = dict(self.cfg) #must not put 'self' in the lambda, but this is ok
        target = lambda infile: _process_interlopers(infile, **target_kwargs)
        all_out_arrays = pool.map(target, self.infiles)

        _log('INTERLOPER REDUCTION')
        all_out_arrays = reduce(lambda a, b: a + b, all_out_arrays)
        listdict_merge = lambda d1, d2: {
            d1.keys()[0]: dict(
                (
                    k, 
                    d1[d1.keys()[0]].get(k, []) + d2[d2.keys()[0]].get(k, [])
                ) for k in set(d1[d1.keys()[0]]).union(d2[d2.keys()[0]])
            )
        }
        unique_keys = np.unique([aoa.keys()[0] for aoa in all_out_arrays])
        for progress, k in enumerate(unique_keys):
            _log(' ', progress, '/', len(unique_keys))
            kis = np.flatnonzero(np.array([aoa.keys()[0] for aoa in all_out_arrays]) == k)
            all_out_arrays[kis[0]] = reduce(listdict_merge, np.array(all_out_arrays)[kis])
            for ki in kis[1:][::-1]:
                del all_out_arrays[ki]
        all_out_arrays = {k: v for all_out_array in all_out_arrays for k, v in all_out_array.items()}

        _log('INTERLOPER OUTPUT')
        with h5py.File(self.cfg.outfile, 'a') as f:
            for cluster_id, interlopers in all_out_arrays.items():
                self._write_interlopers(f, cluster_id, interlopers)
        
        return

    def orbit_search(self):

        ##Memory usage is too high if collecting everything before write to file.
        ##Solution is to write after processing each file.
        ##But doing this in parallel requires blocking, which I haven't figured out.
        for infile in self.infiles:
            _log('ORBIT SEARCH')
            out_arrays = self._process_orbits(infile, **self.cfg)
            with h5py.File(self.cfg.outfile, 'a') as f:
                _log('ORBIT OUTPUT')
                for progress, out_array in enumerate(out_arrays):
                    if (progress % 1000) == 0:
                        _log(' ', progress, '/', len(out_arrays))
                    self._write_satellite(f, out_array)
            del out_arrays

        _log('END')

        return

    def _find_files(self):

        infiles = [self.cfg.treedir + '/tree_{0:.0f}_{1:.0f}_{2:.0f}.dat'.format(i, j, k) \
                   for i, j, k in itertools.product(
                           range(self.cfg.ndivs),
                           range(self.cfg.ndivs), 
                           range(self.cfg.ndivs)
                   )]
        infiles = [infile for infile in infiles if os.path.exists(infile)]
        self.infiles = infiles[:4]

        return

    def _write_headers(self):
        _log('HEADER OUTPUT')
        with h5py.File(self.cfg.outfile, 'w') as f:
            f.create_group('clusters')
            g = f.create_group('config')
            for key, value in self.cfg.items():
                g.attrs[key] = value

            if self.cfg.skipsnaps > 0:
                g['scales'] = self.cfg.scales[:-self.cfg.skipsnaps]

            else:
                g['scales'] = self.cfg.scales

            g['scales'].attrs['units'] = '-'
            g['scales'].attrs['description'] = 'simulation output scale factors'

        return

    def _write_cluster(self, f, data):

        g = f.create_group('clusters/'+str(data['ids'][-1]))
        g.create_group('satellites')
        g.create_group('interlopers')

        g['ids'] = np.array(data['ids'], dtype=np.int)
        g['ids'].attrs['units'] = '-'
        g['ids'].attrs['description'] = 'halo id'

        g['mvir'] = np.array(data['mvirs'], dtype=np.float)
        g['mvir'].attrs['units'] = 'Msun'
        g['mvir'].attrs['description'] = 'halo virial mass (Bryan & Norman 1998)'

        g['rvir'] = np.array(data['rvirs'], dtype=np.float)
        g['rvir'].attrs['units'] = 'kpc'
        g['rvir'].attrs['description'] = 'halo virial radius (Bryan & Norman 1998)'

        g['vrms'] = np.array(data['vrmss'], dtype=np.float)
        g['vrms'].attrs['units'] = 'km/s'
        g['vrms'].attrs['description'] = 'halo velocity dispersion'

        g['subhalo'] = np.array(data['subhalos'], dtype=np.bool)
        g['subhalo'].attrs['units'] = '-'
        g['subhalo'].attrs['description'] = 'halo is subhalo?'

        g['superparent_mvir'] = np.array(data['superparent_mvirs'], dtype=np.float)
        g['superparent_mvir'].attrs['units'] = 'Msun'
        g['superparent_mvir'].attrs['description'] = 'if halo is subhalo, ultimate parent virial mass'

        g['xyz'] = np.vstack((
            np.array(data['xs'], dtype=np.float),
            np.array(data['ys'], dtype=np.float),
            np.array(data['zs'], dtype=np.float)
        )).T
        g['xyz'].attrs['units'] = 'Mpc'
        g['xyz'].attrs['description'] = 'halo coordinates (x, y, z)'

        g['vxyz'] = np.vstack((
            np.array(data['vxs'], dtype=np.float),
            np.array(data['vys'], dtype=np.float),
            np.array(data['vzs'], dtype=np.float)
        )).T
        g['vxyz'].attrs['units'] = 'km/s'
        g['vxyz'].attrs['description'] = 'halo velocity (vx, vy, vz)'

        return

    def _write_interlopers(self, f, cluster_id, data):

        g = f['clusters/'+str(cluster_id)+'/interlopers']

        g['ids'] = np.array(data['ids'], dtype=np.int)
        g['ids'].attrs['units'] = '-'
        g['ids'].attrs['description'] = 'halo id'

        g['mvir'] = np.array(data['mvirs'], dtype=np.float)
        g['mvir'].attrs['units'] = 'Msun'
        g['mvir'].attrs['description'] = 'halo virial mass (Bryan & Norman 1998)'

        g['rvir'] = np.array(data['rvirs'], dtype=np.float)
        g['rvir'].attrs['units'] = 'kpc'
        g['rvir'].attrs['description'] = 'halo virial radius (Bryan & Norman 1998)'

        g['vrms'] = np.array(data['vrmss'], dtype=np.float)
        g['vrms'].attrs['units'] = 'km/s'
        g['vrms'].attrs['description'] = 'halo velocity dispersion'

        g['xyz'] = np.vstack((
            np.array(data['xs'], dtype=np.float),
            np.array(data['ys'], dtype=np.float),
            np.array(data['zs'], dtype=np.float)
        )).T
        g['xyz'].attrs['units'] = 'Mpc'
        g['xyz'].attrs['description'] = 'halo coordinates (x, y, z)'

        g['vxyz'] = np.vstack((
            np.array(data['vxs'], dtype=np.float),
            np.array(data['vys'], dtype=np.float),
            np.array(data['vzs'], dtype=np.float)
        )).T
        g['vxyz'].attrs['units'] = 'km/s'
        g['vxyz'].attrs['description'] = 'halo velocity (vx, vy, vz)'

        return

    def _write_satellite(self, f, data):

        g = f['clusters/'+str(data['cluster_id'])+'/satellites'].create_group(str(data['ids'][-1]))

        g['ids'] = np.array(data['ids'], dtype=np.int)
        g['ids'].attrs['units'] = '-'
        g['ids'].attrs['description'] = 'halo id'

        g['mvir'] = np.array(data['mvirs'], dtype=np.float)
        g['mvir'].attrs['units'] = 'Msun'
        g['mvir'].attrs['description'] = 'halo virial mass (Bryan & Norman 1998)'
        g['mvir'].attrs['max(mvir)/Msun'] = str(np.max(data['mvirs']))

        g['rvir'] = np.array(data['rvirs'], dtype=np.float)
        g['rvir'].attrs['units'] = 'kpc'
        g['rvir'].attrs['description'] = 'halo virial radius (Bryan & Norman 1998)'

        g['vrms'] = np.array(data['vrmss'], dtype=np.float)
        g['vrms'].attrs['units'] = 'km/s'
        g['vrms'].attrs['description'] = 'halo velocity dispersion'

        g['xyz'] = np.vstack((
            np.array(data['xs'], dtype=np.float), 
            np.array(data['ys'], dtype=np.float), 
            np.array(data['zs'], dtype=np.float)
        )).T
        g['xyz'].attrs['units'] = 'Mpc'
        g['xyz'].attrs['description'] = 'halo coordinates (x, y, z)'

        g['vxyz'] = np.vstack((
            np.array(data['vxs'], dtype=np.float), 
            np.array(data['vys'], dtype=np.float), 
            np.array(data['vzs'], dtype=np.float)
        )).T
        g['vxyz'].attrs['units'] = 'km/s'
        g['vxyz'].attrs['description'] = 'halo velocity (vx, vy, vz)'

        g['is_subsub'] = np.array(data['subsub'], dtype=np.bool)
        g['is_subsub'].attrs['units'] = '-'
        g['is_subsub'].attrs['description'] = 'halo is a sub-sub-[...]-halo'

        g['sp_is_fpp'] = np.array(data['sp_ids'], dtype=np.int) == \
                         np.array(f['clusters/'+str(data['cluster_id'])+'/ids'])
        g['sp_is_fpp'].attrs['units'] = '-'
        g['sp_is_fpp'].attrs['description'] = 'superparent of halo is primary progenitor '\
                                              'of superparent at final scale'

        g = g.create_group('superparent')

        g['ids'] = np.array(data['sp_ids'], dtype=np.int)
        g['ids'].attrs['units'] = '-'
        g['ids'].attrs['description'] = 'halo id'

        g['mvir'] = np.array(data['sp_mvirs'], dtype=np.float)
        g['mvir'].attrs['units'] = 'Msun'
        g['mvir'].attrs['description'] = 'halo virial mass (Bryan & Norman 1998)'

        g['rvir'] = np.array(data['sp_rvirs'], dtype=np.float)
        g['rvir'].attrs['units'] = 'kpc'
        g['rvir'].attrs['description'] = 'halo virial radius (Bryan & Norman 1998)'

        g['vrms'] = np.array(data['sp_vrmss'], dtype=np.float)
        g['vrms'].attrs['units'] = 'km/s'
        g['vrms'].attrs['description'] = 'halo velocity dispersion'

        g['xyz'] = np.vstack((
            np.array(data['sp_xs'], dtype=np.float), 
            np.array(data['sp_ys'], dtype=np.float), 
            np.array(data['sp_zs'], dtype=np.float)
        )).T
        g['xyz'].attrs['units'] = 'Mpc'
        g['xyz'].attrs['description'] = 'halo coordinates (x, y, z)'

        g['vxyz'] = np.vstack((
            np.array(data['sp_vxs'], dtype=np.float), 
            np.array(data['sp_vys'], dtype=np.float), 
            np.array(data['sp_vzs'], dtype=np.float)
        )).T
        g['vxyz'].attrs['units'] = 'km/s'
        g['vxyz'].attrs['description'] = 'halo velocity (vx, vy, vz)'

        return

#called by parallel function, keep outside class
def _extract_cluster_arrays(cluster_branch, h0=None):

    data = {}

    data['ids'] = [halo.id if halo is not None else -1 for halo in cluster_branch]
    data['mvirs'] = [halo.mvir / h0 if halo is not None else np.nan for halo in cluster_branch]
    data['rvirs'] = [halo.rvir / h0 if halo is not None else np.nan for halo in cluster_branch]
    data['vrmss'] = [halo.vrms if halo is not None else np.nan for halo in cluster_branch]

    subhalos = []
    for halo in cluster_branch:
        if halo is not None:
            if halo.parent is not None:
                subhalos.append(True)
            else:
                subhalos.append(False)
        else:
            subhalos.append(False)
    data['subhalos'] = subhalos

    superparent_mvirs = []
    for halo in cluster_branch:
        superparent = _get_superparent(halo)
        if superparent is not None:
            superparent_mvirs.append(superparent.mvir / h0)
        else:
            superparent_mvirs.append(np.nan)

    data['superparent_mvirs'] = superparent_mvirs

    data['xs'] = [halo.pos[0] / h0 if halo is not None else np.nan for halo in cluster_branch]
    data['ys'] = [halo.pos[1] / h0 if halo is not None else np.nan for halo in cluster_branch]
    data['zs'] = [halo.pos[2] / h0 if halo is not None else np.nan for halo in cluster_branch]

    data['vxs'] = [halo.vel[0] if halo is not None else np.nan for halo in cluster_branch]
    data['vys'] = [halo.vel[1] if halo is not None else np.nan for halo in cluster_branch]
    data['vzs'] = [halo.vel[2] if halo is not None else np.nan for halo in cluster_branch]

    return data

def _extract_interloper_arrays(halo, is_near, h0=None):

    data = {}

    data['ids'] = [halo.id]
    data['mvirs'] = [halo.mvir / h0]
    data['rvirs'] = [halo.rvir / h0]
    data['vrmss'] = [halo.vrms]
    data['xs'] = [halo.pos[0] / h0]
    data['ys'] = [halo.pos[1] / h0]
    data['zs'] = [halo.pos[2] / h0]
    data['vxs'] = [halo.vel[0]]
    data['vys'] = [halo.vel[1]]
    data['vzs'] = [halo.vel[2]]

    return {k: data for k in is_near}

def _extract_orbit_arrays(halo_branch, superparent_branch, h0=None):

    data = {}

    data['cluster_id'] = _get_superparent(halo_branch[-1]).id

    data['ids'] = [halo.id if halo is not None else -1 for halo in halo_branch]
    data['mvirs'] = [halo.mvir / h0 if halo is not None else np.nan for halo in halo_branch]
    data['rvirs'] = [halo.rvir / h0 if halo is not None else np.nan for halo in halo_branch]
    data['vrmss'] = [halo.vrms if halo is not None else np.nan for halo in halo_branch]
    data['xs'] = [halo.pos[0] / h0 if halo is not None else np.nan for halo in halo_branch]
    data['ys'] = [halo.pos[1] / h0 if halo is not None else np.nan for halo in halo_branch]
    data['zs'] = [halo.pos[2] / h0 if halo is not None else np.nan for halo in halo_branch]
    data['vxs'] = [halo.vel[0] if halo is not None else np.nan for halo in halo_branch]
    data['vys'] = [halo.vel[1] if halo is not None else np.nan for halo in halo_branch]
    data['vzs'] = [halo.vel[2] if halo is not None else np.nan for halo in halo_branch]
    data['subsub'] = [
        (_get_superparent(halo).id == halo.parent.id) \
        if _get_superparent(halo) is not None \
        else False \
        for halo in halo_branch
    ]

    data['sp_ids'] = [halo.id if halo is not None else -1 for halo in superparent_branch]
    data['sp_mvirs'] = [
        halo.mvir / h0 if halo is not None else np.nan for halo in superparent_branch
    ]
    data['sp_rvirs'] = [
        halo.rvir / h0 if halo is not None else np.nan for halo in superparent_branch
    ]
    data['sp_vrmss'] = [halo.vrms if halo is not None else np.nan for halo in superparent_branch]
    data['sp_xs'] = [
        halo.pos[0] / h0 if halo is not None else np.nan for halo in superparent_branch
    ]
    data['sp_ys'] = [
        halo.pos[1] / h0 if halo is not None else np.nan for halo in superparent_branch
    ]
    data['sp_zs'] = [
        halo.pos[2] / h0 if halo is not None else np.nan for halo in superparent_branch
    ]
    data['sp_vxs'] = [halo.vel[0] if halo is not None else np.nan for halo in superparent_branch]
    data['sp_vys'] = [halo.vel[1] if halo is not None else np.nan for halo in superparent_branch]
    data['sp_vzs'] = [halo.vel[2] if halo is not None else np.nan for halo in superparent_branch]

    return data

#parallel function: must be outside class, prefer simple arguments
def _process_clusters(infile, scales=None, skipsnaps=None, h0=None, **kwargs):

    _log('  processing file', infile.split('/')[-1])

    read_tree.read_tree(infile)
    all_halos = read_tree.all_halos
    halo_tree = read_tree.halo_tree

    nsnaps = len(scales) - skipsnaps

    out_arrays = []

    for halo in halo_tree.halo_lists[skipsnaps].halos:

        if halo.parent is not None: #centrals only
            continue

        if (halo.mvir / h0 > m_max_cluster.value) or \
           (halo.mvir / h0 < m_min_cluster.value):
            continue

        cluster_branch = []

        for level in range(nsnaps):

            if len(cluster_branch) == 0:
                cluster_branch.append(halo)

            elif cluster_branch[-1] is None:
                cluster_branch.append(None)

            else:
                cluster_branch.append(cluster_branch[-1].prog)

        cluster_branch = cluster_branch[::-1]

        out_arrays.append(_extract_cluster_arrays(cluster_branch, h0=h0))

    read_tree.delete_tree()

    return out_arrays

#parallel function: must be outside class, prefer simple arguments
def _process_interlopers(infile, outfile=None, skipsnaps=None, h0=None, lbox=None,\
                         m_min_satellite=None, m_max_satellite=None, interloper_dR=None,\
                         interloper_dV=None, **kwargs):

    #read clusters here will happen for each parallel process, but would be copied for each 
    #process anyway, would need to explicitly set up a shared memory object to work around this
    #note: placed here it gets destroyed when no longer needed

    cluster_ids = []
    cluster_xyzs = []
    cluster_vzs = []
    cluster_rvirs = []
    cluster_vrmss = []

    with h5py.File(outfile, 'r') as f:

        for cluster_key, cluster in f['clusters'].items():

            cluster_ids.append(cluster['ids'][-1 - skipsnaps])
            cluster_xyzs.append(cluster['xyz'][-1 - skipsnaps])
            cluster_vzs.append(cluster['vxyz'][-1 - skipsnaps, 2])
            cluster_rvirs.append(cluster['rvir'][-1 - skipsnaps])
            cluster_vrmss.append(cluster['vrms'][-1 - skipsnaps])

    cluster_ids = np.array(cluster_ids, dtype=np.long)
    cluster_xyzs = np.array(cluster_xyzs, dtype=np.float)
    cluster_vzs = np.array(cluster_vzs, dtype=np.float)
    cluster_rvirs = np.array(cluster_rvirs, dtype=np.float)
    cluster_vrmss = np.array(cluster_vrmss, dtype=np.float)

    _log('  processing file', infile.split('/')[-1])

    read_tree.read_tree(infile)
    all_halos = read_tree.all_halos
    halo_tree = read_tree.halo_tree

    out_arrays = []

    for halo in halo_tree.halo_lists[skipsnaps].halos:

        if (halo.mvir / h0 > m_max_satellite.value) or \
           (halo.mvir / h0 < m_min_satellite.value):
            continue

        xyz = np.array([
            halo.pos[0] / h0, 
            halo.pos[1] / h0, 
            halo.pos[2] / h0
        ], dtype=np.float)
        vz = np.array([halo.vel[2]], dtype=np.float)

        D = xyz - cluster_xyzs
        D[D > lbox.value / 2.] -= lbox.value
        D[D < -lbox.value / 2.] += lbox.value
        dvz = np.abs(vz - cluster_vzs + 100.0 * h0 * D[:, 2]) / cluster_vrmss
        D *= 1.E3 / cluster_rvirs[:, np.newaxis] #rvir in kpc
        D = np.power(D, 2)

        is_near = cluster_ids[
            np.logical_and(
                np.logical_and(
                    D[:, 0] + D[:, 1] < np.power(interloper_dR, 2), #inside of circle
                    np.sum(D, axis=1) > np.power(interloper_dR, 2) #outside of sphere
                ),
                dvz < interloper_dV #inside velocity offset limit
            )
        ]

        if len(is_near):
            out_arrays.append(_extract_interloper_arrays(halo, is_near, h0=h0))

    read_tree.delete_tree()

    return out_arrays

#parallel function: must be outside class, prefer simple arguments
def _process_orbits(infile, skipsnaps=None, h0=None, lbox=None, m_min_satellite=None,\
                    m_max_satellite=None, m_min_cluster=None, m_max_cluster=None, **kwargs):

    _log('  processing file', infile.split('/')[-1])

    read_tree.read_tree(infile)
    all_halos = read_tree.all_halos
    halo_tree = read_tree.halo_tree

    nsnaps = len(scales) - skipsnaps

    out_arrays = []

    for halo in halo_tree.halo_lists[skipsnaps].halos:

        if (halo.mvir / h0 > m_max_satellite.value) or \
           (halo.mvir / h0 < m_min_satellite.value):
            continue

        superparent = _get_superparent(halo)

        if superparent is None:
            continue

        if (superparent.mvir / h0 < m_min_cluster.value) or \
           (superparent.mvir / h0 > m_max_cluster.value):
            continue

        halo_branch = []
        superparent_branch = []

        for level in range(nsnaps):

            if len(halo_branch) == 0:
                halo_branch.append(halo)
                superparent_branch.append(_get_superparent(halo))

            elif halo_branch[-1] is None:
                halo_branch.append(None)
                superparent_branch.append(None)

            else:
                halo_branch.append(halo_branch[-1].prog)
                superparent_branch.append(_get_superparent(halo_branch[-1]))

        halo_branch = halo_branch[::-1]
        superparent_branch = superparent_branch[::-1]

        out_arrays.append(_extract_orbit_arrays(halo_branch, superparent_branch, h0=h0))

    read_tree.delete_tree()

    return out_arrays

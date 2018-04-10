import numpy as np
import h5py
from joblib import Parallel, delayed
import itertools
import os
import read_tree
from sys import argv
from datetime import datetime
if len(argv) < 2:
    print 'Provide config file as argument.'
    exit()
execfile(argv[1])

def timer():
    T = datetime.now()
    return ' [{0:02d}:{1:02d}:{2:02d}]'.format(T.hour, T.minute, T.second)

def find_files():

    maxdivs = 20
    infiles = []
    
    for i, j, k in itertools.product(range(maxdivs), range(maxdivs), range(maxdivs)):
        fname = cfg.treedir + '/tree_' + str(i) + '_' + str(j) + '_' + str(k) + '.dat'
        if os.path.exists(fname):
            infiles.append(fname)
            
    return infiles

def get_superparent(halo):

    if halo is None:
        return None
    
    if halo.parent is None:
        return None

    retval = halo.parent
    while retval.parent is not None:
        retval = retval.parent

    return retval

def extract_cluster_arrays(cluster_branch):

    data = {}
    
    data['ids'] = [halo.id if halo is not None else -1 for halo in cluster_branch]

    data['mvirs'] = [halo.mvir / cfg.h0 if halo is not None else np.nan for halo in cluster_branch]
    
    data['rvirs'] = [halo.rvir / cfg.h0 if halo is not None else np.nan for halo in cluster_branch]

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
        superparent = get_superparent(halo)
        if superparent is not None:
            superparent_mvirs.append(superparent.mvir / cfg.h0)
        else:
            superparent_mvirs.append(np.nan)
                
    data['superparent_mvirs'] = superparent_mvirs

    data['xs'] = [halo.pos[0] / cfg.h0 if halo is not None else np.nan for halo in cluster_branch]
    data['ys'] = [halo.pos[1] / cfg.h0 if halo is not None else np.nan for halo in cluster_branch]
    data['zs'] = [halo.pos[2] / cfg.h0 if halo is not None else np.nan for halo in cluster_branch]
    
    data['vxs'] = [halo.vel[0] if halo is not None else np.nan for halo in cluster_branch]
    data['vys'] = [halo.vel[1] if halo is not None else np.nan for halo in cluster_branch]
    data['vzs'] = [halo.vel[2] if halo is not None else np.nan for halo in cluster_branch]

    return data

def extract_interloper_arrays(halo, is_near):

    data = {}

    data['ids'] = [halo.id]
    data['mvirs'] = [halo.mvir / cfg.h0]
    data['rvirs'] = [halo.rvir / cfg.h0]
    data['vrmss'] = [halo.vrms]
    data['xs'] = [halo.pos[0] / cfg.h0]
    data['ys'] = [halo.pos[1] / cfg.h0]
    data['zs'] = [halo.pos[2] / cfg.h0]
    data['vxs'] = [halo.vel[0]]
    data['vys'] = [halo.vel[1]]
    data['vzs'] = [halo.vel[2]]
    
    return {k: data for k in is_near}

def extract_orbit_arrays(halo_branch, superparent_branch):

    data = {}

    data['cluster_id'] = get_superparent(halo_branch[-1]).id
    
    data['ids'] = [halo.id if halo is not None else -1 for halo in halo_branch]
    data['mvirs'] = [halo.mvir / cfg.h0 if halo is not None else np.nan for halo in halo_branch]
    data['rvirs'] = [halo.rvir / cfg.h0 if halo is not None else np.nan for halo in halo_branch]
    data['vrmss'] = [halo.vrms if halo is not None else np.nan for halo in halo_branch]
    data['xs'] = [halo.pos[0] / cfg.h0 if halo is not None else np.nan for halo in halo_branch]
    data['ys'] = [halo.pos[1] / cfg.h0 if halo is not None else np.nan for halo in halo_branch]
    data['zs'] = [halo.pos[2] / cfg.h0 if halo is not None else np.nan for halo in halo_branch]
    data['vxs'] = [halo.vel[0] if halo is not None else np.nan for halo in halo_branch]
    data['vys'] = [halo.vel[1] if halo is not None else np.nan for halo in halo_branch]
    data['vzs'] = [halo.vel[2] if halo is not None else np.nan for halo in halo_branch]
    data['subsub'] = [(get_superparent(halo).id == halo.parent.id) if get_superparent(halo) is not None else False for halo in halo_branch]

    data['sp_ids'] = [halo.id if halo is not None else -1 for halo in superparent_branch]
    data['sp_mvirs'] = [halo.mvir / cfg.h0 if halo is not None else np.nan for halo in superparent_branch]
    data['sp_rvirs'] = [halo.rvir / cfg.h0 if halo is not None else np.nan for halo in superparent_branch]
    data['sp_vrmss'] = [halo.vrms if halo is not None else np.nan for halo in superparent_branch]
    data['sp_xs'] = [halo.pos[0] / cfg.h0 if halo is not None else np.nan for halo in superparent_branch]
    data['sp_ys'] = [halo.pos[1] / cfg.h0 if halo is not None else np.nan for halo in superparent_branch]
    data['sp_zs'] = [halo.pos[2] / cfg.h0 if halo is not None else np.nan for halo in superparent_branch]
    data['sp_vxs'] = [halo.vel[0] if halo is not None else np.nan for halo in superparent_branch]
    data['sp_vys'] = [halo.vel[1] if halo is not None else np.nan for halo in superparent_branch]
    data['sp_vzs'] = [halo.vel[2] if halo is not None else np.nan for halo in superparent_branch]
    
    return data

def write_headers(f):

    f.create_group('clusters')
    g = f.create_group('config')
    for key, value in cfg.items():
        g.attrs.create(key, value)
    #g.attrs.create(r'dimensionless hubble constant (cfg.h0)', cfg.h0)
    #g.attrs.create(r'minimum cluster mass [Msun] (cfg.m_min_cluster)', cfg.m_min_cluster)
    #g.attrs.create(r'maximum cluster mass [Msun] (cfg.m_max_cluster)', cfg.m_max_cluster)
    #g.attrs.create(r'minimum satellite mass [Msun] (cfg.m_min_satellite)', cfg.m_min_satellite)
    #g.attrs.create(r'maximum satellite mass [Msun] (cfg.m_max_satellite)', cfg.m_max_satellite)
    #g.attrs.create(r'box side length [Mpc] (cfg.lbox)', cfg.lbox)
    #g.attrs.create(r'radial interloper search length [host Rvir] (cfg.interloper_dR)', cfg.interloper_dR)
    #g.attrs.create(r'velocity interloper search length [host vrms] (cfg.interloper_dV)', cfg.interloper_dV)

    if cfg.skipsnaps > 0:
        g['scales'] = scales[:-cfg.skipsnaps]

    else:
        g['scales'] = scales

    g['scales'].attrs.create(r'units', r'-')
    g['scales'].attrs.create(r'description', r'simulation output scale factors')

    return

def write_cluster(f, data):
        
    g = f.create_group('clusters/'+str(data['ids'][-1]))
    g.create_group('satellites')
    g.create_group('interlopers')

    g['ids'] = np.array(data['ids'], dtype=np.int)
    g['ids'].attrs.create(r'units', r'-')
    g['ids'].attrs.create(r'description', r'halo id')

    g['mvir'] = np.array(data['mvirs'], dtype=np.float)
    g['mvir'].attrs.create(r'units', r'Msun')
    g['mvir'].attrs.create(r'description', r'halo virial mass (Bryan & Norman 1998)')

    g['rvir'] = np.array(data['rvirs'], dtype=np.float)
    g['rvir'].attrs.create(r'units', r'kpc')
    g['rvir'].attrs.create(r'description', r'halo virial radius (Bryan & Norman 1998)')

    g['vrms'] = np.array(data['vrmss'], dtype=np.float)
    g['vrms'].attrs.create(r'units', r'km/s')
    g['vrms'].attrs.create(r'description', r'halo velocity dispersion')

    g['subhalo'] = np.array(data['subhalos'], dtype=np.bool)
    g['subhalo'].attrs.create(r'units', '-')
    g['subhalo'].attrs.create(r'description', r'halo is subhalo?')

    g['superparent_mvir'] = np.array(data['superparent_mvirs'], dtype=np.float)
    g['superparent_mvir'].attrs.create(r'units', r'Msun')
    g['superparent_mvir'].attrs.create(r'description', r'if halo is subhalo, ultimate parent virial mass')

    g['xyz'] = np.vstack((np.array(data['xs'], dtype=np.float), np.array(data['ys'], dtype=np.float), np.array(data['zs'], dtype=np.float))).T
    g['xyz'].attrs.create(r'units', r'Mpc')
    g['xyz'].attrs.create(r'description', r'halo coordinates (x, y, z)')

    g['vxyz'] = np.vstack((np.array(data['vxs'], dtype=np.float), np.array(data['vys'], dtype=np.float), np.array(data['vzs'], dtype=np.float))).T
    g['vxyz'].attrs.create(r'units', r'km/s')
    g['vxyz'].attrs.create(r'description', r'halo velocity (vx, vy, vz)')

    return

def write_interlopers(f, cluster_id, data):

    g = f['clusters/'+str(cluster_id)+'/interlopers']

    g['ids'] = np.array(data['ids'], dtype=np.int)
    g['ids'].attrs.create(r'units', r'-')
    g['ids'].attrs.create(r'description', r'halo id')
    
    g['mvir'] = np.array(data['mvirs'], dtype=np.float)
    g['mvir'].attrs.create(r'units', r'Msun')
    g['mvir'].attrs.create(r'description', r'halo virial mass (Bryan & Norman 1998)')

    g['rvir'] = np.array(data['rvirs'], dtype=np.float)
    g['rvir'].attrs.create(r'units', r'kpc')
    g['rvir'].attrs.create(r'description', r'halo virial radius (Bryan & Norman 1998)')

    g['vrms'] = np.array(data['vrmss'], dtype=np.float)
    g['vrms'].attrs.create(r'units', r'km/s')
    g['vrms'].attrs.create(r'description', r'halo velocity dispersion')

    g['xyz'] = np.vstack((np.array(data['xs'], dtype=np.float), np.array(data['ys'], dtype=np.float), np.array(data['zs'], dtype=np.float))).T
    g['xyz'].attrs.create(r'units', r'Mpc')
    g['xyz'].attrs.create(r'description', r'halo coordinates (x, y, z)')

    g['vxyz'] = np.vstack((np.array(data['vxs'], dtype=np.float), np.array(data['vys'], dtype=np.float), np.array(data['vzs'], dtype=np.float))).T
    g['vxyz'].attrs.create(r'units', r'km/s')
    g['vxyz'].attrs.create(r'description', r'halo velocity (vx, vy, vz)')

    return

def write_satellite(f, data):

    g = f['clusters/'+str(data['cluster_id'])+'/satellites'].create_group(str(data['ids'][-1]))

    g['ids'] = np.array(data['ids'], dtype=np.int)
    g['ids'].attrs.create(r'units', r'-')
    g['ids'].attrs.create(r'description', r'halo id')
    
    g['mvir'] = np.array(data['mvirs'], dtype=np.float)
    g['mvir'].attrs.create(r'units', r'Msun')
    g['mvir'].attrs.create(r'description', r'halo virial mass (Bryan & Norman 1998)')
    g['mvir'].attrs.create(r'max(mvir)/Msun', str(np.max(data['mvirs'])))

    g['rvir'] = np.array(data['rvirs'], dtype=np.float)
    g['rvir'].attrs.create(r'units', r'kpc')
    g['rvir'].attrs.create(r'description', r'halo virial radius (Bryan & Norman 1998)')

    g['vrms'] = np.array(data['vrmss'], dtype=np.float)
    g['vrms'].attrs.create(r'units', r'km/s')
    g['vrms'].attrs.create(r'description', r'halo velocity dispersion')

    g['xyz'] = np.vstack((np.array(data['xs'], dtype=np.float), np.array(data['ys'], dtype=np.float), np.array(data['zs'], dtype=np.float))).T
    g['xyz'].attrs.create(r'units', r'Mpc')
    g['xyz'].attrs.create(r'description', r'halo coordinates (x, y, z)')

    g['vxyz'] = np.vstack((np.array(data['vxs'], dtype=np.float), np.array(data['vys'], dtype=np.float), np.array(data['vzs'], dtype=np.float))).T
    g['vxyz'].attrs.create(r'units', r'km/s')
    g['vxyz'].attrs.create(r'description', r'halo velocity (vx, vy, vz)')

    g['is_subsub'] = np.array(data['subsub'], dtype=np.bool)
    g['is_subsub'].attrs.create(r'units', r'-')
    g['is_subsub'].attrs.create(r'description', r'halo is a sub-sub-[...]-halo')

    g['sp_is_fpp'] = np.array(data['sp_ids'], dtype=np.int) == np.array(f['clusters/'+str(data['cluster_id'])+'/ids'])
    g['sp_is_fpp'].attrs.create(r'units', r'-')
    g['sp_is_fpp'].attrs.create(r'description', r'superparent of halo is primary progenitor of superparent at final scale')

    g = g.create_group('superparent')

    g['ids'] = np.array(data['sp_ids'], dtype=np.int)
    g['ids'].attrs.create(r'units', r'-')
    g['ids'].attrs.create(r'description', r'halo id')
    
    g['mvir'] = np.array(data['sp_mvirs'], dtype=np.float)
    g['mvir'].attrs.create(r'units', r'Msun')
    g['mvir'].attrs.create(r'description', r'halo virial mass (Bryan & Norman 1998)')

    g['rvir'] = np.array(data['sp_rvirs'], dtype=np.float)
    g['rvir'].attrs.create(r'units', r'kpc')
    g['rvir'].attrs.create(r'description', r'halo virial radius (Bryan & Norman 1998)')

    g['vrms'] = np.array(data['sp_vrmss'], dtype=np.float)
    g['vrms'].attrs.create(r'units', r'km/s')
    g['vrms'].attrs.create(r'description', r'halo velocity dispersion')

    g['xyz'] = np.vstack((np.array(data['sp_xs'], dtype=np.float), np.array(data['sp_ys'], dtype=np.float), np.array(data['sp_zs'], dtype=np.float))).T
    g['xyz'].attrs.create(r'units', r'Mpc')
    g['xyz'].attrs.create(r'description', r'halo coordinates (x, y, z)')

    g['vxyz'] = np.vstack((np.array(data['sp_vxs'], dtype=np.float), np.array(data['sp_vys'], dtype=np.float), np.array(data['sp_vzs'], dtype=np.float))).T
    g['vxyz'].attrs.create(r'units', r'km/s')
    g['vxyz'].attrs.create(r'description', r'halo velocity (vx, vy, vz)')
    
    return

def process_clusters(infile):

    print '  processing file', infile.split('/')[-1], timer()

    read_tree.read_tree(infile)
    all_halos = read_tree.all_halos
    halo_tree = read_tree.halo_tree

    nsnaps = len(scales) - cfg.skipsnaps

    out_arrays = []

    for halo in halo_tree.halo_lists[cfg.skipsnaps].halos:
        
        if halo.parent is not None: #centrals only
            continue

        if (halo.mvir / cfg.h0 > cfg.m_max_cluster) or (halo.mvir / cfg.h0 < cfg.m_min_cluster):
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
        
        out_arrays.append(extract_cluster_arrays(cluster_branch))
            
    read_tree.delete_tree()

    return out_arrays

def process_interlopers(infile):

    #read clusters here will happen for each parallel process, but would be copied for each process anyway, would need to explicitly set up a shared memory object to work around this
    #note: placed here it gets destroyed when no longer needed
    
    cluster_ids = []
    cluster_xyzs = []
    cluster_vzs = []
    cluster_rvirs = []
    cluster_vrmss = []

    with h5py.File(cfg.outfile, 'r') as f:

        for cluster_key, cluster in f['clusters'].items():

            cluster_ids.append(cluster['ids'][-1 - cfg.skipsnaps])
            cluster_xyzs.append(cluster['xyz'][-1 - cfg.skipsnaps])
            cluster_vzs.append(cluster['vxyz'][-1 - cfg.skipsnaps, 2])
            cluster_rvirs.append(cluster['rvir'][-1 - cfg.skipsnaps])
            cluster_vrmss.append(cluster['vrms'][-1 - cfg.skipsnaps])

    cluster_ids = np.array(cluster_ids, dtype=np.long)
    cluster_xyzs = np.array(cluster_xyzs, dtype=np.float)
    cluster_vzs = np.array(cluster_vzs, dtype=np.float)
    cluster_rvirs = np.array(cluster_rvirs, dtype=np.float)
    cluster_vrmss = np.array(cluster_vrmss, dtype=np.float)
    
    print '  processing file', infile.split('/')[-1], timer()

    read_tree.read_tree(infile)
    all_halos = read_tree.all_halos
    halo_tree = read_tree.halo_tree

    out_arrays = []

    for halo in halo_tree.halo_lists[cfg.skipsnaps].halos:

        if (halo.mvir / cfg.h0 > cfg.m_max_satellite) or (halo.mvir / cfg.h0 < cfg.m_min_satellite):
            continue

        xyz = np.array([halo.pos[0] / cfg.h0, halo.pos[1] / cfg.h0, halo.pos[2] / cfg.h0], dtype=np.float)
        vz = np.array([halo.vel[2]], dtype=np.float)

        D = xyz - cluster_xyzs
        D[D > cfg.lbox / 2.] -= cfg.lbox
        D[D < -cfg.lbox / 2.] += cfg.lbox
        dvz = np.abs(vz - cluster_vzs + 100.0 * cfg.h0 * D[:, 2]) / cluster_vrmss
        D *= 1.E3 / cluster_rvirs[:, np.newaxis] #rvir in kpc
        D = np.power(D, 2)

        is_near = cluster_ids[
            np.logical_and(
                np.logical_and(
                    D[:, 0] + D[:, 1] < np.power(cfg.interloper_dR, 2), #inside circular aperture
                    np.sum(D, axis=1) > np.power(cfg.interloper_dR, 2) #and outside spherical aperture
                ),
                dvz < cfg.interloper_dV #and inside velocity offset limit
            )
        ]

        if len(is_near):
            out_arrays.append(extract_interloper_arrays(halo, is_near))
        
    read_tree.delete_tree()
    
    return out_arrays

def process_orbits(infile):

    print '  processing file', infile.split('/')[-1], timer()

    read_tree.read_tree(infile)
    all_halos = read_tree.all_halos
    halo_tree = read_tree.halo_tree

    nsnaps = len(scales) - cfg.skipsnaps

    out_arrays = []

    for halo in halo_tree.halo_lists[cfg.skipsnaps].halos:

        if (halo.mvir / cfg.h0 > cfg.m_max_satellite) or (halo.mvir / cfg.h0 < cfg.m_min_satellite):
            continue

        superparent = get_superparent(halo)

        if superparent is None:
            continue

        if (superparent.mvir / cfg.h0 < cfg.m_min_cluster) or (superparent.mvir / cfg.h0 > cfg.m_max_cluster):
            continue

        halo_branch = []
        superparent_branch = []

        for level in range(nsnaps):

            if len(halo_branch) == 0:
                halo_branch.append(halo)
                superparent_branch.append(get_superparent(halo))

            elif halo_branch[-1] is None:
                halo_branch.append(None)
                superparent_branch.append(None)

            else:
                halo_branch.append(halo_branch[-1].prog)
                superparent_branch.append(get_superparent(halo_branch[-1]))

        halo_branch = halo_branch[::-1]
        superparent_branch = superparent_branch[::-1]
        
        out_arrays.append(extract_orbit_arrays(halo_branch, superparent_branch))
        
    read_tree.delete_tree()
    
    return out_arrays



#MAIN PROGRAM START
        
print 'START' + timer()
scales = np.loadtxt(cfg.scalefile, unpack=True, usecols=[1])

infiles = find_files()

#WRITE HEADERS

print 'HEADER OUTPUT' + timer()
with h5py.File(cfg.outfile, 'w') as f:
    write_headers(f)
        
#FIND CLUSTERS
        
print 'CLUSTER SEARCH' + timer()
all_out_arrays = Parallel(n_jobs=min(cfg.ncpu, len(infiles)))(delayed(process_clusters)(infile) for infile in infiles)
print 'CLUSTER REDUCTION' + timer()
all_out_arrays = reduce(lambda a, b: a + b, all_out_arrays)

#WRITE CLUSTERS

print 'CLUSTER OUTPUT' + timer()
with h5py.File(cfg.outfile, 'a') as f:
    for out_array in all_out_arrays:
        write_cluster(f, out_array)

#FIND INTERLOPERS

print 'INTERLOPER SEARCH' + timer()
all_out_arrays = Parallel(n_jobs=min(cfg.ncpu, len(infiles)))(delayed(process_interlopers)(infile) for infile in infiles)

print 'INTERLOPER REDUCTION' + timer()
all_out_arrays = reduce(lambda a, b: a + b, all_out_arrays)
listdict_merge = lambda d1, d2: {d1.keys()[0]: dict((k, d1[d1.keys()[0]].get(k, []) + d2[d2.keys()[0]].get(k, [])) for k in set(d1[d1.keys()[0]]).union(d2[d2.keys()[0]]))}
unique_keys = np.unique([aoa.keys()[0] for aoa in all_out_arrays])
for progress, k in enumerate(unique_keys):
    print ' ', progress, '/', len(unique_keys), timer()
    kis = np.flatnonzero(np.array([aoa.keys()[0] for aoa in all_out_arrays]) == k)
    all_out_arrays[kis[0]] = reduce(listdict_merge, np.array(all_out_arrays)[kis])
    for ki in kis[1:][::-1]:
        del all_out_arrays[ki]
all_out_arrays = {k: v for all_out_array in all_out_arrays for k, v in all_out_array.items()}

#WRITE INTERLOPERS

print 'INTERLOPER OUTPUT' + timer()
with h5py.File(cfg.outfile, 'a') as f:
    for cluster_id, interlopers in all_out_arrays.items():
        write_interlopers(f, cluster_id, interlopers)

#FIND ORBITS

#print 'ORBIT SEARCH' + timer()
#Parallel breaks because Pickle is used to pass return values, and pickle protocol 2 breaks for data >~2GB in size -> go to python 3.5+ with protocol 4 to fix this, but python 3 not on champion
#all_out_arrays = Parallel(n_jobs=min(cfg.ncpu, len(infiles)))(delayed(process_orbits)(infile) for infile in infiles)

#this also tends to run out of memory, so clear buffer between each input file
#all_out_arrays = [process_orbits(infile) for infile in infiles] #serial mode
#print 'ORBIT REDUCTION' + timer()
#all_out_arrays = reduce(lambda a, b: a + b, all_out_arrays)

        

#WRITE ORBITS

#print 'ORBIT OUTPUT' + timer()
#with h5py.File(cfg.outfile, 'a') as f:
#    for progress, out_array in enumerate(all_out_arrays):
#        if (progress % 1000) == 0:
#            print ' ', progress, '/', len(all_out_arrays), timer()
#        write_satellite(f, out_array)

for infile in infiles:
    print 'ORBIT SEARCH' + timer()
    out_arrays = process_orbits(infile)
    with h5py.File(cfg.outfile, 'a') as f:
        print 'ORBIT OUTPUT' + timer()
        for progress, out_array in enumerate(out_arrays):
            if (progress % 1000) == 0:
                print ' ', progress, '/', len(out_arrays)
            write_satellite(f, out_array)
    del out_arrays

print 'END' + timer()


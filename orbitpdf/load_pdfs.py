import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from multiprocessing import Lock
import h5py

_lock = Lock()


def _tailor_j(js, rank=None, pdffile=None, intervals=None, Qkey=None,
              Qbins=None, use_interlopers=None, use_Mmax=None):
    PDFs = np.ones((len(js), (len(Qbins) - 1))) * np.nan
    with _lock:
        with h5py.File(pdffile) as f:
            Mhost_s = np.log10(f['/satellites/Mhost'][()])
            if use_Mmax:
                Msat_s = np.log10(f['/satellites/m_max'][()])
            else:
                Msat_s = np.log10(f['/satellites/Msat'][()])
            V_s = f['/satellites/V'][()]
            R_s = f['/satellites/R'][()]
            Q_s = f['/satellites/{:s}'.format(Qkey)][()]
            Mhost_i = np.log10(f['/interlopers/Mhost'][()])
            Msat_i = np.log10(f['/interlopers/Msat'][()])
            V_i = f['/interlopers/V'][()]
            R_i = f['/interlopers/R'][()]
    for j, (Mhost_j, Msat_j, V_j, R_j) in enumerate(js):
        mask_s = (np.abs(Mhost_s - Mhost_j) < intervals[0]) \
            * (np.abs(Msat_s - Msat_j) < intervals[1]) \
            * (np.abs(R_s - R_j) < intervals[2]) \
            * (np.abs(V_s - V_j) < intervals[3])
        N_s = np.sum(mask_s)
        if N_s > 0:
            mask_i = (np.abs(Mhost_i - Mhost_j) < intervals[0]) \
                * (np.abs(Msat_i - Msat_j) < intervals[1]) \
                * (np.abs(R_i - R_j) < intervals[2]) \
                * (np.abs(V_i - V_j) < intervals[3])
            N_i = np.sum(mask_i)
            norm = N_s + N_i if use_interlopers else N_s
            Q_j = Q_s[mask_s]
            PDF = np.histogram(Q_j, bins=Qbins)[0]
            PDFs[j] = PDF / norm
        else:
            PDFs[j] = np.zeros(len(Qbins) - 1)
    return PDFs


def load_pdfs(
        pdffile,
        Mhost,
        Msat,
        R,
        V,
        intervals=(.5, .5, .05, .04),
        Qkey='t_peri',
        Qbins=np.linspace(0, 2, 100),
        use_interlopers=True,
        use_Mmax=True,
        ncpu=2
):
    '''
    Construct pdf for an aperture in (Mhost, Msat, R, V).

    Parameters
    ----------
    pdffile : str
        Path to an hdf5 file containing orbital pdfs.

    Mhost : array_like
        Bin centre(s) in host mass, in Msun.

    Msat : array_like
        Bin centre(s) in satellite total (not stellar) mass, in Msun.

    V : array_like
        Bin centre(s) in normalized velocity coordinates v_LoS/sig3D.

    R : array_like
        Bin centre(s) in normalized radial coordinate r_proj/rvir.

    intervals : iterable of length 4
        Bin half-widths in (Mhost, Msat, R, V). The Mhost and Msat
        intervals should be given in dex, while the R and V intervals
        are simply in normalized units. (Default: (.5, .5, .05, .04).)

    Qkey : str
        Orbital parameter to be binned. Currently supported:
        - 'm_infall': mass at infall time in Msun.
        - 'm_max': maximum mass at any time (past or present) in Msun.
        - 'r': current deprojected radius in units of rvir.
        - 'r_min': minimum deprojected radius at any time (past or present) in
          units of rvir.
        - 't_infall': infall time across 2.5rvir, provided as a scale factor.
        - 't_peri': (DEFAULT) pericentre time, provided as a scale factor.
        - 'v': current (deprojected) speed in units of sigma3D.
        - 'v_max': maximum (deprojected) speed at any time (past or present) in
          units of sigma3D.

    Qbins : array_like
        Bin edges to use for the PDF.

    use_interlopers : bool
        Whether to include interlopers in the calculation. If True, the
        returned PDF will not sum to 1; the balance is the interloper
        probability. (Default: True.)

    use_Mmax : bool
        If True, the values used for Msat will be the maximum (past) masses
        of the satellite haloes. If False, their current masses are used
        instead. (Default: False.)

    ncpu : int
        Number of processors to use for parallel calculation. (Default: 2.)
    '''
    Mhost = np.log10(Mhost)
    Msat = np.log10(Msat)

    js = np.array_split(
        np.array(
            list(zip(Mhost, Msat, V, R)),
            dtype=np.dtype([
                ('Mhost', np.float),
                ('Msat', np.float),
                ('V', np.float),
                ('R', np.float),
            ])
        ),
        ncpu
    )
    ranked_js = list(zip(range(ncpu), js))

    target_kwargs = dict(
        pdffile=pdffile,
        intervals=intervals,
        Qkey=Qkey,
        Qbins=Qbins,
        use_interlopers=use_interlopers,
        use_Mmax=use_Mmax
    )

    def target(rj):
        rank, j = rj
        return _tailor_j(j, rank=rank, **target_kwargs)

    with Pool(ncpu) as pool:
        PDFs = pool.map(target, ranked_js)
    return np.vstack(PDFs)

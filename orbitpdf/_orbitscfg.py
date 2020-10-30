import astropy.units as U


class OrbitsConfig(dict):

    reqkeys = {
        'h0',
        'm_min_cluster',
        'm_max_cluster',
        'm_min_satellite',
        'm_max_satellite',
        'lbox',
        'interloper_dR',
        'interloper_dV',
        'treedir',
        'scalefile',
        'H',
        'z'
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value
        return

    def _validate(self):

        MAXDIVS = 20

        if set(self.keys()) < self.reqkeys:
            raise AttributeError("Missing OrbitsConfig attributes "
                                 "(check OrbitsConfig.reqkeys).")
        if 'skipsnaps' not in self.keys():
            self.skipsnaps = 0
        if 'skipmore_for_select' not in self.keys():
            self.skipmore_for_select = 0
        if 'ncpu' not in self.keys():
            self.ncpu = 1
        if 'ndivs' not in self.keys():
            self.ndivs = MAXDIVS
        if 'outfile' not in self.keys():
            self.outfile = 'orbits_out.hdf5'

        try:
            self['m_min_cluster'] = self['m_min_cluster'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsConfig: Provide units for "
                                 "m_min_cluster (astropy.units).")
        try:
            self['m_max_cluster'] = self['m_max_cluster'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsConfig: Provide units for "
                                 "m_max_cluster (astropy.units).")
        try:
            self['m_min_satellite'] = self['m_min_satellite'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsConfig: Provide units for "
                                 "m_min_satellite (astropy.units).")
        try:
            self['m_max_satellite'] = self['m_max_satellite'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsConfig: Provide units for "
                                 "m_max_satellite (astropy.units).")
        try:
            self['lbox'] = self['lbox'].to(U.Mpc)
        except AttributeError:
            raise AttributeError("OrbitsConfig: Provide units for lbox "
                                 "(astropy.units).")

        return

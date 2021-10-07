from ._orbitscfg import OrbitsConfig
import astropy.units as U


class OrbitPDFConfig(OrbitsConfig):

    reqkeys = {
        'orbitfile',
        'pdf_m_min_satellite',
        'pdf_m_max_satellite',
        'pdf_m_min_cluster',
        'pdf_m_max_cluster',
        'resolution_cut',
        'H',
        'z'
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

        super()._validate()

        if set(self.keys()) < self.reqkeys:
            raise AttributeError("Missing OrbitPDFConfig attributes "
                                 "(check OrbitPDFConfig.reqkeys).")
        if 'pdfsfile' not in self.keys():
            self.pdfsfile = 'pdfs_out.hdf5'
        if 'signed_V' not in self.keys():
            self.signed_V = False

        try:
            self['pdf_m_min_satellite'] = self['pdf_m_min_satellite'].to(
                U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for "
                                 "pdf_m_min_satellite (astropy.units).")
        try:
            self['pdf_m_max_satellite'] = self['pdf_m_max_satellite'].to(
                U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for "
                                 "pdf_m_max_satellite (astropy.units).")
        try:
            self['pdf_m_min_cluster'] = self['pdf_m_min_cluster'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for "
                                 "pdf_m_min_cluster (astropy.units).")
        try:
            self['pdf_m_max_cluster'] = self['pdf_m_max_cluster'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for "
                                 "pdf_m_max_cluster (astropy.units).")
        try:
            self['resolution_cut'] = self['resolution_cut'].to(U.Msun)
        except AttributeError:
            raise AttributeError("OrbitsPDFConfig: Provide units for "
                                 "resolution_cut (astropy.units).")

        return

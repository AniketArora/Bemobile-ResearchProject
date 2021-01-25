"""TrafficJamPrediction main module entry point."""

from .basemap import Basemap


class JamPrediction(object):
    """
        [OBSOLETE]
        JamPrediction module is currently not used.
        This might be used in the future to wrap other modules.
    """
    
    def __init__(self, *args, **kwargs):
        
        self.basemap = {}
        
        if 'basemap' in kwargs:
            self.basemap = kwargs['basemap']
    
    @property
    def basemap(self):
        return self.__basemap
    
    @basemap.setter
    def basemap(self, basemap_data):
        # TODO: Add support to switch between giving a Basemap or only Basemap_data
        # TODO: If that support cannot be guaranteed, only allow the user to set a Basmeap.
        self.__basemap = Basemap(basemap_data)
            
            

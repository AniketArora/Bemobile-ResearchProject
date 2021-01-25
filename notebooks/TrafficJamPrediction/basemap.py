"""TrafficJamPrediction all things related to the Basemap."""

import pandas as pd
import logging


class Basemap(object):
    """The Basemap class is used for all things related to the Basemap.
    - Checking whether the Basemap contains a specific part.
    - Loading in CSV files.
    - Loading in a whole directory.


    """    
    
    
    def __init__(self, basemap_data, *args, **kwargs):
        self.basemap = basemap_data
        
    @property
    def basemap(self):
        return self.__basemap
    
    @basemap.setter
    def basemap(self, basemap_data):
        self.__basemap = basemap_data
    
    @property
    def map(self):
        return self.__basemap
    
        
    def __str__(self):        
        return str(self.basemap)
    
    def __repr__(self):
        return f"<Basemap: {str(self.basemap.keys())}>"

    def check(self, *parts):        
        """Check whether a certain key exist in the basemap.
        This can be used to assert that the Basemap contains the required part.

        Args:
            *parts: List of required parts to check for in the Basemap.
        Raises:
            TypeError: If the basemap does not contain a specific part, a TypeError will be raised.

        Returns:
            True: Returns True if all the required parts are included in the Basemap.
        """        
        for part in parts:
            if part not in self.basemap.keys():
                raise TypeError(f"This basemap does not contain {part}")

        return True
        
    def load_csv(self, name: str, location: str, index: str = None):  
        """Load a specific CSV file into the Basemap.
        A new dictionary entry will be created with a Pandas DataFrame created from the CSV file.

        * This method is also wrapped in the Basemap.load() method

        Args:
            name (str, optional): A one-word name for the Dictionary entry. E.g.: "Segments".
            location (str, optional): The location where the CSV file is located.
            index (str, optional): The index to set for the dataframe in the Dictionary entry. Defaults to None for no Index.
        """           
        self.basemap[name] = pd.read_csv(location,sep=';')
        if index is not None:
            self.basemap[name].set_index(index, inplace=True)
        
    def load(self, parts: dict, directory = None, *args, **kwargs):
        """Load a whole directory of CSV's for the Basemap

        Args:
            parts (dict): A dictionary of the parts to include in the Basemap. \nThe keys of this dictionary are the names for the entries in the basemap. The value of one dictionary entry is again a dictionary with the Index and/or Location that's necessary.
            E.g.: {
                'segments': {'index': 'SegmentID'},
                'nodes': {'location': 'Bemobile.Basemap/nodes'}, # No .csv suffix needed.
                }

        Raises:
            ValueError: if a directory parameter is not given, or if it has not been initialized before, a ValueError is raised.
        """               
        if directory is not None and '_directory' in self.basemap:
            self.basemap['_directory'] = kwargs['_directory']
        elif directory is None and '_directory' in self.basemap:
            directory = self.basemap['_directory']
        elif directory is None and '_directory' not in self.basemap:
            pass
        else:
            raise ValueError("The directory was not set during initialization or as a parameter in this function.")

        
        for part, options in parts.items():
            location = options.get('location', part) # If the location is given, use that, otherwise use the name to get the location.
            index = options.get('index', None) # If the index is given, use that.
            self.load_csv(name = part, index = index , location=f"{directory}/{location}.csv")
        

    def save_to_cache(self):
        """
            Saves the basemap as Pickled dataframes
        """
        raise NotImplementedError("Still to implement")

    def load_from_cache(self):
        """
            Loads the basemap from Pickled dataframes
        """
        raise NotImplementedError("Still to implement")
        
    
            
    
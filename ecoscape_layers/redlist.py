import requests
from ebird.api import get_taxonomy

class RedList():
    """
    A module of functions that primarily involve interfacing with the IUCN Red List API.
    """

    def __init__(self, redlist_key: str):
        """
        Initializes a RedList object.
        API keys are required to access the IUCN Red List API and eBird API respectively; see the documentation for more information.
        """
        self.redlist_params = { "token": redlist_key }

    def get_from_redlist(self, url):
        """
        Convenience function for sending GET request to Red List API with the key.

        :param url: the URL for the request.
        :return: response for the request.
        """
        res = requests.get(url, params=self.redlist_params).json()
        return res["result"]

    def get_habitat_data(self, name, region=None):
        """
        Gets habitat assessments for suitability for a given species.
        This also adds the associated landcover map's code and resistance value to the API response, which are useful for creating resistance mappings and/or habitat layers.

        :param name: scientific name of the species.
        :param region: a specific region to assess habitats in (see https://apiv3.iucnredlist.org/api/v3/docs#regions).
        :return: a list of habitats identified by the IUCN Red List as suitable for the species.
        """
        url = "https://apiv3.iucnredlist.org/api/v3/habitats/species/name/{0}".format(name)
        if region is not None:
            url += "/region/{1}".format(region)

        habs = self.get_from_redlist(url)

        for hab in habs:
            code = hab["code"]
            sep = code.index(".")
            # only take up to level 2 (xx.xx), therefore truncating codes with more than 1 period separator
            if code.count(".") > 1:
                code = code[:code.index(".", sep+1)]
            hab["map_code"] = int(code[:sep] + code[sep+1:].zfill(2))
            hab["resistance"] = 0 if hab["majorimportance"] == "Yes" else 0.1

        return habs

    def get_elevation(self, name):
        '''
        Obtain elevation bounds that are suitable for a given species
        :param name: scientific name of the species
        '''
        url = "https://apiv3.iucnredlist.org/api/v3/species/{0}".format(name)
        res = self.get_from_redlist(url)

        if len(res) == 0:
            return -10000, 10000
        else:
            # if elevation_lower is None, assume -10000; if elevation_upper is None, assume 10000
            return res[0]["elevation_lower"] or -10000, res[0]["elevation_upper"] or 10000

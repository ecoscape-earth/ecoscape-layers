import requests
from ebird.api import get_taxonomy


class RedList:
    """
    A module of functions that primarily involve interfacing with the IUCN Red List API.
    """

    def __init__(self, redlist_key: str, ebird_key: str | None = None):
        """
        Initializes a RedList object.
        API keys are required to access the IUCN Red List API and eBird API respectively; see the documentation for more information.
        """
        self.redlist_params = {"token": redlist_key}
        self.ebird_key = ebird_key

    def get_from_redlist(self, url: str) -> dict:
        """
        Convenience function for sending GET request to Red List API with the key.

        :param url: the URL for the request.
        :return: response for the request.
        """
        res = requests.get(url, params=self.redlist_params)

        if res.status_code != 200:
            raise ValueError(f"Error {res.status_code} in Red List API request")

        data: dict = res.json()

        return data["result"]

    def get_scientific_name(self, species_code: str) -> str:
        """Translates eBird codes to scientific names for use in Red List.

        Args:
            species_code (str): eBird code for a bird species.

        Raises:
            ValueError: When the eBird API is not provided.
            ValueError: When the provided species_code is not found with eBird API

        Returns:
            str: The scientific name of the bird species
        """

        if self.ebird_key is None:
            raise ValueError("eBird API key is required to get scientific names")

        # Manual corrections here for differences between eBird and IUCN Red List scientific names.
        # This should probably be changed in the future.
        if species_code == "whhwoo":
            return "Leuconotopicus albolarvatus"
        elif species_code == "yebmag":
            return "Pica nutalli"
        elif species_code == "pilwoo":
            return "Hylatomus pileatus"
        elif species_code == "recwoo":
            return "Leuconotopicus borealis"

        res = get_taxonomy(self.ebird_key, species=species_code)
        sci_name = res[0]["sciName"] if len(res) > 0 else None

        if sci_name is None:
            error = f"sci_name for eBird code '{species_code}' could not be found."
            raise ValueError(error)

        return sci_name

    def get_habitat_data(
        self, species_name: str, region=None, ebird_code: bool = False
    ) -> dict[int, dict[str, str | bool]]:
        """Gets habitat assessments for suitability for a given species.
        This also adds the associated landcover/terrain map's code to the API response,
        which is useful for creating resistance mappings and/or habitat layers.

        Args:
            species_name (str): scientific name of the species.
            region (_type_, optional): a specific region to assess habitats in (see https://apiv3.iucnredlist.org/api/v3/docs#regions).. Defaults to None.
            ebird_code (bool, optional): If True, reads species_name as an eBird species_code and converts it to a scientific/iucn name. Defaults to False.

        Raises:
            ValueError: Errors when the code received from the IUCN Redlist is missing a period or data after a period.
            KeyError: If the IUCN Redlist has two types of habitat with the

        Returns:
            dict[int, dict[str, str | bool]]: a dictionary map codes which reference habitats identified by the IUCN Red List as suitable for the species.
        """

        # Check if species_name needs to be converted to sci_name in the case of using ebird's species_code
        if ebird_code:
            sci_name = self.get_scientific_name(species_name)
        else:
            sci_name = species_name

        url = f"https://apiv3.iucnredlist.org/api/v3/habitats/species/name/{sci_name}"
        if region is not None:
            url += f"/region/{region}"

        habs = self.get_from_redlist(url)

        for hab in habs:
            code = str(hab["code"])

            # some codes are in the format xx.xx.xx instead of xx.xx
            # we will truncate xx.xx.xx codes to xx.xx
            code_sep = code.split(".")

            # check that code_sep len is not less than len of 2
            if len(code_sep) < 2:
                error = f"The code '{code}' is missing the required number of '.'"
                raise ValueError(error)

            # take a sub-array of the first two elements
            code_sep = code_sep[:2]

            # fill each number in two have length two so format is [xx, xx]
            code_sep = map(lambda num_str: num_str.zfill(2), code_sep)

            # create a map_code that is represented by an int
            hab["map_code"] = int("".join(code_sep))

            # Convert bool like strings to bools
            hab["majorimportance"] = (hab["majorimportance"] == "Yes")
            hab["suitability"] = (hab["suitability"] == "Suitable")

        # transform to dict with keys as map_codes and values as hab data
        res = {}
        for hab in habs:
            # save the map_code and delete it from the dict
            map_code: int = hab["map_code"]
            del hab["map_code"]

            if map_code in res:
                # this should not happen as long as the RedList API is consistent
                raise KeyError(f"map_code {map_code} was found multiple times")
            else:
                res[map_code] = hab

        return res

    def get_elevation(self, name) -> tuple[int, int]:
        """
        Obtain elevation bounds that are suitable for a given species
        :param name: scientific name of the species
        """
        url = f"https://apiv3.iucnredlist.org/api/v3/species/{name}"
        res = self.get_from_redlist(url)

        if len(res) == 0:
            return -10000, 10000
        else:
            # if elevation_lower is None, assume -10000; if elevation_upper is None, assume 10000
            return (
                int(res[0]["elevation_lower"]) or -10000,
                int(res[0]["elevation_upper"]) or 10000,
            )

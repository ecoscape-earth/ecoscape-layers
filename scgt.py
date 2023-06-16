import rasterio
import rasterio.plot
import shutil
from rasterio.windows import Window, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from osgeo import gdal
import scipy.ndimage as nd

class GeoTiff(object):
    def __init__(self, file=None):
        """initialize a GeoTiff object

        properties:
        dataset : open geotiff file
        size : touple -> raster width, height
        bands : int -> number of bands in geotiff
        transform : the dataset's geospatial transform - an affine transformation matrix that maps pixel
                    locations in (row, col) coordinates to (x, y) spatial positions
        corners : array of lat/long of corners, in order top left, top right, bottom right, bottom left
        block_shapes : array with the shape of blocks for all bands
                        i.e [(1, (3, 791)), (2, (3, 791))]
        profile : Geotiff profile - used for writing metadata to new file
        """
        if file:
            self.dataset = file
            self.filepath = file.name
            self.size = (self.dataset.width, self.dataset.height)
            self.width, self.height = self.dataset.width, self.dataset.height
            self.bands = self.dataset.count
            self.transform = self.dataset.transform
            self.corners = [self.transform * (0, 0),                                    # top left
                            self.transform * (0, self.dataset.height),                  # top right
                            self.transform * (self.dataset.width, self.dataset.height), #bottom right
                            self.transform * (self.dataset.width, 0)]                   #bottom left
            self.block_shapes = self.dataset.block_shapes
            self.data_types = self.dataset.dtypes
            self.profile = self.dataset.profile
            self.crs = self.dataset.crs


    def __enter__(self):
        # essentially returns what to use for the variable in a with statement
        return self

    def __exit__(self, type, value, traceback):
        # closes the dataset at the conclusion of a with statement
        self.getAttributeTable()
        self.dataset.close()

    @classmethod
    def from_file(cls, filename):
        # creates a new GeoTiff obj from given geotiff file
        # opens in rw if file exists, else creates new file and opens in append mode
        open_file = rasterio.open(filename, 'r+')
        if not open_file:
            sys.exit("GeoTiff Error: from_file() being called with invalid Geotiff file")
        # return GeoTiff obj with open file
        return cls(open_file)

    @classmethod
    def copy_to_new_file(cls, filename, profile, no_data_value=None):
        """ creates a new file to store empty GeoTiff obj with specified params
        :param no_data_value (dtype of raster): value to be used as transparent 'nodata' value, otherwise fills 0's
            (note that if geotiff is of unsigned type, like uint8, the no_data value must be  a positive int in the data 
            range, which could result in data obstruction. If datatype is signed, we suggest using a negative value)
        """
        # create file with write mode, then open with rw to have full read/write access
        f = rasterio.open(filename, 'w', **profile)
        f.close()
        # sets no_data (transparent value)
        if no_data_value is not None:
            with rasterio.open(filename, "r+") as dataset:
                dataset.nodata = no_data_value
                nodata_mask = np.ones((dataset.width, dataset.height)) * no_data_value
                dataset.write(nodata_mask.astype(profile['dtype']), 1)

        open_file = rasterio.open(filename, 'r+', **profile)
        if not open_file:
            sys.exit("GeoTiff Error: copy_to_new_file() being called with invalid Geotiff data or filepath")
        # return GeoTiff obj with open file
        return cls(open_file)


    def clone_shape(self, filepath, no_data_value=None, dtype=None):
        """Creates a new geotiff with the indicated filepath.
        :param no_data_value (dtype of raster): value to be used as transparent 'nodata' value, otherwise fills 0's
            (note that if geotiff is of unsigned type, like uint8, the no_data value must be  a positive int in the data 
            range, which could result in data obstruction. If datatype is signed, we suggest using a negative value)
        :param dtype (str): datatype of tiff (ref: https://github.com/rasterio/rasterio/blob/master/rasterio/dtypes.py#L21)
        """
        profile = self.profile
        # check that datatype is valid
        if dtype is not None:
            assert rasterio.dtypes.check_dtype(dtype), \
                f"The datatype {dtype} is not recognized by rasterio"
            profile['dtype'] = dtype
        # check that no_data_value has same datatype as raster
        if no_data_value is not None:
            assert rasterio.dtypes.can_cast_dtype(no_data_value, profile['dtype']), \
                f"The chosen no_data value {no_data_value} cannot be cast to type {profile['dtype']} without loss of information"
        
        # copies src tif file to destination
        shutil.copy(src=self.filepath, dst=filepath)
        tiff = GeoTiff.copy_to_new_file(filepath, profile=profile, no_data_value=no_data_value)
        return tiff

    def scale_tiff(self, reference_tif=None, scale_x=1, scale_y=1):
        # scales given geotiff to the same dims as reference, or by scalar factor
        if reference_tif is not None:
            scale_x = reference_tif.width/self.width
            scale_y = reference_tif.height/self.height

        self.dataset.close()
        with rasterio.open(self.filepath) as dataset:
            profile = dataset.profile.copy()
            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * scale_y),
                    int(dataset.width * scale_x)
                ),
                resampling=Resampling.bilinear
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (1 / scale_x),
                (1 / scale_y)
            )
            profile.update({"height": data.shape[-2],
                            "width": data.shape[-1],
                        "transform": transform})

        open_file = rasterio.open(self.filepath, 'r+', **profile)
        if not open_file:
            sys.exit("GeoTiff Error: copy_to_new_file() being called with invalid Geotiff data or filepath")
        open_file.write(data)


    def get_reader(self, b=0, w=None, h=None):
        #Returns a reader with the specified, w, h, b.
        return Reader(self, b, w, h)

    def set_tile(self, tile):
        # check that tile dimensions are within geotiff bounds, if not alter
        if tile.x + tile.w > self.width:
            tile.w = self.width - tile.x
        if tile.y + tile.h > self.height:
            tile.h = self.height - tile.y
        # Converts the type to the appropriate type for the output.
        if isinstance(tile.m, np.ndarray):
            T = tile.m.astype(self.dataset.dtypes[0])
        else:
            T = tile.m.numpy().astype(self.dataset.dtypes[0])
        # Adjust T if border exists
        if tile.b > 0:
            b = tile.b
            if tile.x + tile.w == self.width:
                T = T[:, :, b:]
            elif tile.x == 0:
                T = T[:, :, :-b]
            else:
                T = T[:, :, b:-b]
            if tile.y + tile.h == self.height:
                T = T[:, b:, :]
            elif tile.y == 0:
                T = T[:, :-b, :]
            else:
                T = T[:, b:-b, :]
        # Sets geotiff's data to the values of the tile.
        self.dataset.write(T, window=tile.get_window(includes_border=False))

    def get_pixel_from_coord(self, coord):
        """
        Gets pixel on the geotiff that corresponds to the CRS coordinate given
        :param coord: coordinate to get (already in geotiff's CRS)
        returns: (x,y) pixel of geotiff which corresponds to coord, or None if out of bounds
        """
        x,y = coord
        y, x = self.dataset.index(x, y)
        # check that coord is within bounds of tif
        w, h = self.size
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        return x, y

    def get_tile_from_coord(self, coord, tile_scale = 4):
        """
        gets tile of size tile_scale x tile_scale centered at coord from geotiff
        :param coord: coordinate to get (already in geotiff's CRS)
        :param tile_scale: size of the tile
            (if the pixel resolution is 300m, and tile_scale is 3, the tile will be 900m x 900m centered around coord)
        :returns: numpy array representation of the tile, or None if coord is out of range
        """
        xy = self.get_pixel_from_coord(coord)
        if xy is None:
            return None
        x, y = xy
        tile_x = x - (tile_scale/2) if x - (tile_scale/2) > 0 else 0
        tile_y = y - (tile_scale/2) if y - (tile_scale/2) > 0 else 0
        np_tile = self.get_tile(tile_scale, tile_scale, 0, tile_x, tile_y)
        return np_tile

    def set_tile_from_coord(self, coord, value, tile_scale = 4):
        """
        sets the geotiff to value at the given coord
        """
        """
        gets tile of size tile_scale x tile_scale centered at coord from geotiff
        :param geotiff (scgt.GeoTiff): geotiff to get the tile from
        :param coord: coordinate to get (already in geotiff's CRS)
        :param tile_scale: size of the tile 
            (if the pixel resolution is 300m, and tile_scale is 3, the tile will be 900m x 900m centered around coord)
        :returns: numpy array representation of the tile, or None if coord is out of range
        """
        # precondition: coord in geotiff range
        lat, lon = coord
        xy = self.get_pixel_from_coord((lat, lon))
        if xy is None:
            return None
        x, y = xy
        tile_x = x - (tile_scale/2) if x - (tile_scale/2) > 0 else 0
        tile_y = y - (tile_scale/2) if y - (tile_scale/2) > 0 else 0
        vals = np.full((1, tile_scale, tile_scale), value)
        tile = Tile(tile_scale, tile_scale, 0, self.bands, tile_x, tile_y, vals)
        self.set_tile(tile)

    def get_tile(self, w, h, b, x, y):
        """
        Returns a tile at a certain position with border b
        reads this window from all bands in Geotiff storing in 3D np array
        of dimensions (2b+w, 2b+h, bands)
        """
        window = Window(x - b, y - b, w + 2*b, h + 2*b)
        arr = self.dataset.read(window=window)
        tile = Tile(w, h, b, self.bands, x, y, arr)
        return tile

    def get_all_as_tile(self):
        """Gets the entire content of the geotiff as a tile with empty border."""
        window = Window(0, 0, self.width, self.height)
        arr = self.dataset.read(window=window)
        return Tile(self.width, self.height, 0, self.bands, 0, 0, arr)

    def get_tile_from_window(self, w, border):
        # creates tile at given window with border b
        window_w_border = Window(w.col_off - border, w.row_off - border, w.width + 2*border, w.height + 2* border)
        arr = self.dataset.read(window=window_w_border)
        tile = Tile(w.width, w.height, border, self.bands, w.col_off, w.row_off, arr)
        return tile

    def get_rectangle(self, range_width, range_height, band=None):
        """
        reads any square of geotiff x1:x2, y1:y2 into a numpy matrix
        returns a 3D numpy matrix, with dimensions (x1 - x0, y1 - y0, c),
        where c is the number of channels (bands) of the geotiff.
        unless layer is specified, then the result is a 2D matrix of shape (x1 - x0, y1 - y0).
        """
        x0, x1 = range_width
        y0, y1 = range_height
        width, height = x1-x0, y1-y0
        # if no layer specified, get rectangle from every layer and append to array
        if band is None:
            m = np.empty(shape=(width, height, self.bands))
            for band in range(1, self.bands):
                window = Window(x0, y0, width, height)
                arr = self.dataset.read(band, window=window)
                m.append(arr)
        else:
            window = Window(x0, y0, width, height)
            m = self.dataset.read(band, window=window)
        return m

    def file_write(self, filename):
        """
        Writes itself to a filename, using inherent blocking of geotiff to write
        """
        shutil.copy(src=self.filepath, dst=filename)

    def getAttributeTable(self):
        """
        Opens geotiff with GDAL, creates attribute table containing tiff's unique values and counts.
        To be called in __exit__ as to update the RAT with any new values
        """

        # Get unique values and counts in the band
        uniqueCounts = {}
        reader = self.get_reader(b=0, w=10000, h=10000)

        for tile in reader:
            tile.fit_to_bounds(width=self.width, height=self.height)
            window = Window(tile.x, tile.y, tile.w, tile.h)

            t_unique, t_counts = np.unique(self.dataset.read(window=window), return_counts=True)
            for i in range(len(t_unique)):
                uniqueCounts[t_unique[i]] = uniqueCounts.get(t_unique[i], 0) + t_counts[i]

        # https://gdal.org/python/osgeo.gdal.RasterAttributeTable-class.html
        # https://gdal.org/python/osgeo.gdalconst-module.html
        ds = gdal.Open(self.filepath)
        rb = ds.GetRasterBand(1)

        # Create and populate the RAT
        rat = gdal.RasterAttributeTable()

        rat.CreateColumn('VALUE', gdal.GFT_Integer, gdal.GFU_Generic)
        rat.CreateColumn('COUNT', gdal.GFT_Real, gdal.GFU_Generic)

        for i, (unique, count) in enumerate(uniqueCounts.items()):
            rat.SetValueAsInt(i, 0, int(unique))
            rat.SetValueAsDouble(i, 1, float(count))

        # Associate with the band
        rb.SetDefaultRAT(rat)

        # Close the dataset and persist the RAT
        ds = None

    def draw_geotiff(self, width=5, height=5, band=1):
        """Plots tile's numpy array representation at given band."""
        plt.figure(figsize=(width, height))
        fig, ax = plt.subplots(figsize = (width, height))
        fig.colorbar(ax.imshow(self.get_rectangle((0, self.width), (0, self.height), band),
                                                  cmap="inferno"))
        plt.show()

    def crop_to_new_file(self, output_path, bounds, padding=0):
        """
        Create a new geotiff by cropping the current one and writing to a new file.
        int padding: amount of padding in meters to add around the shape bounds
        tuple bounds: bounding box (xmin, ymin, xmax, ymax) for the output (in the same coordinate system)
        """
        # add padding to the bounds
        bounds = (bounds[0] - padding, bounds[1] - padding, bounds[2] + padding, bounds[3] + padding)

        # keep window within bounds of self, and round lengths and offsets to keep windows aligned
        src_window = from_bounds(*self.dataset.bounds, transform=self.dataset.transform)
        cropped_window = from_bounds(*bounds, transform=self.dataset.transform).intersection(src_window).round_lengths().round_offsets(pixel_precision=0)
        x_offset, y_offset = cropped_window.col_off, cropped_window.row_off

        # update metadata for new file based on the main window of interest
        profile = self.dataset.profile
        profile.update({
            'width': cropped_window.width,
            'height': cropped_window.height,
            'transform': self.dataset.window_transform(cropped_window),
            'nodata': None,
            'compress': 'LZW',
            'bigtiff': 'YES'
        })

        # copy data within the cropping window over to new file
        output = GeoTiff.copy_to_new_file(output_path, profile)
        reader = output.get_reader(b=0, w=10000, h=10000)
        for tile in reader:
            tile.fit_to_bounds(width=output.width, height=output.height)
            window = Window(tile.x, tile.y, tile.w, tile.h)
            src_window = Window(tile.x + x_offset, tile.y + y_offset, tile.w, tile.h)
            output.dataset.write(self.dataset.read(window=src_window), window=window)

        return output

    def reproject_from_crs(self, output_path, dest_crs, res=None, resampling="nearest"):
        """
        Reprojects geotiff to a new CRS. A specific resolution given as a tuple (xSize, ySize) can be specified.
        See https://rasterio.readthedocs.io/en/latest/topics/reproject.html.
        """
        resampling = getattr(Resampling, resampling, None)
        if resampling is None:
            resampling = Resampling.nearest
            print('Invalid resampling arg, using nearest_neighbor')

        transform, width, height = calculate_default_transform(self.dataset.crs, dest_crs, self.width, self.height, *self.dataset.bounds, resolution=res)

        kwargs = self.dataset.meta.copy()
        kwargs.update({
            'crs': dest_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'LZW',
            'bigtiff': 'YES'
        })

        with GeoTiff.copy_to_new_file(output_path, kwargs) as dest:
            for i in range(1, self.dataset.count + 1):
                reproject(
                    source=rasterio.band(self.dataset, i),
                    destination=rasterio.band(dest.dataset, i),
                    src_transform=self.dataset.transform,
                    src_crs=self.dataset.crs,
                    dst_transform=transform,
                    dst_crs=dest_crs,
                    resampling=resampling)

    def reproject_from_tiff(self, output_path, ref_tiff, resampling="near"):
        """
        Reprojects geotiff to the CRS/resolution of a reference tiff. In addition, this ensures that the tiffs are aligned exactly with each other.
        See https://gdal.org/programs/gdalwarp.html and https://gdal.org/api/python/osgeo.gdal.html.
        """

        ds = gdal.Open(self.filepath)

        kwargs = {
            'dstSRS': ref_tiff.dataset.crs,
            'xRes': ref_tiff.dataset.transform[0],
            'yRes': ref_tiff.dataset.transform[4],
            'outputBounds': ref_tiff.dataset.bounds,
            'overviewLevel': None,
            'creationOptions': ['COMPRESS=LZW', 'BIGTIFF=YES'],
            'resampleAlg': resampling
        }

        gdal.Warp(output_path, ds, **kwargs)
        ds = None

    def get_pixel_value(self, x, y):
        # window = rasterio.windows.Window(x, y, 1, 1)
        # return (self.dataset.read(window=window))[0][0][0]
        tif_tile = self.get_all_as_tile()
        return np.squeeze(tif_tile.m)[y][x]

    def get_average_around_pixel(self, x, y, size):
        """
        gets weighted average pixel value centered around given pixel
        """
        def gaussian_kernel(asize):
            k = np.zeros((asize, asize))
            j = asize // 2
            k[j, j] = 1
            return nd.gaussian_filter(k, sigma=asize / 4)
        window = rasterio.windows.Window(x - (size/2), y - (size/2), size, size)
        arr = np.squeeze(self.dataset.read(window=window))
        return np.average(arr, weights=gaussian_kernel(arr.shape[0]))

class Reader(object):
    # You may wonder why we care about a reader, rather than just calling
    # get_tile many times.  The problem is that a geotiff can be huge, so a
    # reader should try to leave it on disk and just cache the portion of it
    # that we need each time.

    def __init__(self, geo, b=0, w=None, h=None):
        # dimensions optional: if none given reader will default to natural block dimenstions
        self.geo = geo  # GeoTiff object
        self.b = b      # window border
        self.w = w      # window width
        self.h = h      # window height
        # if w, h not defined set it to block size
        if self.w is None or self.h is None:
            self.w, self.h = self.geo.block_shapes[0]

        self.tile_space_dims = (math.ceil(geo.width/self.w),
                                math.ceil(geo.height/self.h))  # Dimensions for the Tile matrix (Tiles/row, Tiles/col)


        # Iterator state variables

    # Returns an iterator for Reader
    def __iter__(self):
        self.tiles_read = 0       # Int: Tiles read
        self.tile_corner = np.array([0, 0]) # Tuple (x, y): Upper-left corner coordinates of the current Tile
        return self

    # Read and return the next tile
    def __next__(self):
        # Get x,y coordinates for the current Tile
        x,y = self.tile_corner
        # Check if all tiles have been traversed
        if x > self.geo.width or y > self.geo.height:
            raise StopIteration
        tile = self.geo.get_tile(w=self.w, h=self.h, b=self.b, x=x, y=y)

        # Update iterator states
        self.tiles_read += 1
        self.tile_corner += (self.w, 0)
        # Update corner if out of bounds
        if self.tile_corner[0] > self.geo.width:
            self.tile_corner += (-self.tile_corner[0], self.h)

        return tile#Tile(self.w, self.h, self.b, 1, 0,0, tile)

    # Getters
    # Returns the number of tiles read
    def get_tiles_read(self):
        return self.tiles_read
    # Returns the total number of Tiles in object geo
    def get_tile_total(self):
        return self.tile_space_dims[0] * self.tile_space_dims[1]
    # Returns Tile matrix dimensions
    def get_tile_dims(self):
        return self.tile_space_dims



class Tile(object):
    def __init__(self, w, h, b, c, x, y, m):
        self.w = w # width
        self.h = h # height
        self.b = b # border
        self.c = c # number of channels
        self.x = x
        self.y = y
        self.m = m # m is a (2b + w) x (2b + h) x c numpy matrix which contains
                   # the actual data.

    def get_window(self, includes_border=True):
        # gets Window representation of Tile
        # by default, Window does not include border
        if includes_border:
            window=Window(self.x-self.b, self.y-self.b, 2*self.b+self.w, 2*self.b+self.h)
        else:
            window=Window(self.x, self.y, self.w, self.h)
        return window

    def draw_tile(self, width=5, height=5, band=0):
        # plots tile's numpy array representation at given band
        plt.figure(figsize=(width,height))
        plt.imshow(self.m[band])
        plt.show()

    def fit_to_bounds(self, width, height):
        # if tile dimensions are not within width/height bounds, alter them
        if self.x + self.w > width:
            self.w = width - self.x
        if self.y + self.h > height:
            self.h = height - self.y
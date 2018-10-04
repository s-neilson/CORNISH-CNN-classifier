# CORNISH-CNN-classifier
A convolutional neural network for the classification of astronomical obejcts using data that is part of the CORNISH (Co-Ordinated Radio 'N' Infrared Survey for High-mass star formation) project (website: http://cornish.leeds.ac.uk/public/index.php).



# File downloader configuration (downloadConfiguration.txt)
All file paths are relative to the location of the file downloader. Configuration names and their values are seperated by an equal sign.
* objectTypeFilePath: Path to CSV file contains pairs of object names and their associated type.
* outputFileLocation: Path to the folder containing all of the downloaded files; this folder will have subfolders for every object type.
* fileSourceURLPrefix: URL prefix for downloaded files; full download URL contains the prefix followed by the object name followed by a suffix depending on the specific file associated with each object.
* downloadPartialObjects: If equal to "yes" (without quotes), as many files for each object will be downloaded if possible. If equal to anything else, an object will be skipped if one or more of it's files cannot be downloaded.
* downloadFileSuffix: Suffixes of file URLs following the name. There can be many of these suffixes in the download configuration file; each entry of downloadFile Suffix needs to be on a seperate line.

An example of a file URL to download with a fileSourceURLPrefix of http://cornish.leeds.ac.uk/public/data_src/, a name of G009.9702-00.5292 and a downloadFileSuffix of _CORNISH_5GHz.fits is http://cornish.leeds.ac.uk/public/data_src/G009.9702-00.5292_CORNISH_5GHz.fits

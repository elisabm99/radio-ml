import numpy as np
from pytictoc import TicToc
import natpy as nat
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import argparse
import warnings
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from random import uniform
from astropy.coordinates import Angle
import sys

warnings.filterwarnings("ignore")
timer = TicToc ()


# Parse command line arguments
parser = argparse.ArgumentParser(description="Flux spectrum parameters")
parser.add_argument('--alpha', nargs=1, type=float, default=[0])
parser.add_argument('--S0', nargs=1, type=float, default=[1E-4])
parser.add_argument('--i', nargs=1, type=int, default=[100000])
parser.add_argument('--crazy_test', action='store_true')
parser.add_argument('--no-crazy_test', dest='feature', action='store_false')
parser.add_argument('--random_fn', action='store_true')
parser.add_argument('--no-random_fn', dest='feature', action='store_false')
parser.add_argument('--dont_sample_flux', action='store_true')
parser.add_argument('--no-dont_sample_flux', dest='feature', action='store_false')
parser.add_argument('--change_functional_form', action='store_true')
parser.add_argument('--no-change_functional_form', dest='feature', action='store_false')
parser.add_argument('--alt_functional_form', nargs=1, type=int, default=[0]) # 1,2,3 are implemented

# ! Read in arguments
args = parser.parse_args()
print("Arguments", args)
#################################################################################
#################################################################################
#################################################################################





# Catalog frequency
nu = 0.94  # in GHz. Frequency at which to create the catalog. Must be within .78MHz - 1GHz

lognunu0 = np.log(.78 / 1)  # useful log for determination of the spectral index in the bin from .78MHz to 1GHz
# The argument is the ratio of the two bin frequencies considered: 780 MHz and 1 GHz. 
# Spectral index \alpha =  \log( S(0.78 GHz)/S(1 GHz) ) /  lognunu0 

# Flux limit, this is the range of fluxes for sources that go in the catalog
# flux_min = 1e-7  # in Jy
# flux_max = 1e-2  # in Jy
flux_min = 1e-6  # in Jy
flux_max = 1e-2  # in Jy


Npixel = 5120 #2560  # pixel in one direction
image_side = 5  # in degrees. length of one side of the image
pixel_side = image_side / Npixel
# Model sources smaller than 1/3*pixel size as points
resolution_limit = 1. / 3 * pixel_side  # in degrees

# Image parameters
center_ra = 315  # center right ascension of the image in degrees
center_dec = -55  # center declination of the image in degrees
cdeltx = -pixel_side
cdelty = pixel_side
crpix = 2561#1281

# number of sources in TRecs medium catalog
Nagn_trecs = 1461510
Nsfg_trecs = 28263241
N_trecsmedi = Nagn_trecs + Nsfg_trecs
# we want to keep these proportions between AGNs and SFGs
frac_agn = Nagn_trecs / N_trecsmedi
frac_sfg = Nsfg_trecs / N_trecsmedi
# print(frac_agn, frac_sfg)


# Parameters for label creation
i_first_pixel = 1
Npixel_subimage = 128 # number of pixels in the side of the sub image
Nstripes = int(Npixel / Npixel_subimage)  # 16 #32 # each side in divided into Nstripes parts
Nsquares = Nstripes ** 2
Nbins = 10  # number of labels
# This is the range of fluxes for sources that go in the labels
flux_min_label = 1e-6  # Jy
flux_max_label = 1e-4  # Jy

# Flux modification parameters
alpha = args.alpha
S0 = args.S0  # Jy
i = args.i[0]  # an index to label files
crazy_test = args.crazy_test
random_fn = args.random_fn
dont_sample_flux = args.dont_sample_flux
print("dont_sample_flux =", dont_sample_flux)
change_functional_form =args.change_functional_form
alt_functional_form = args.alt_functional_form[0]





# Header of catalog
header_str = "#RA\t\t\t\tDec\t\t\t\tflux@%.2fGHz\t\talpha\t\t\tbeta\t\tMajorAxis\t\tMinorAxis\t       posAngle\n" % nu

# Input/output files
catalog_dir = 'catalogs/'
npzs_dir = 'npzs/'

# Input
file_agn = catalog_dir + 'agnsmedi.dat'
file_sfg = catalog_dir + 'sfgsmedi.dat'
file_agn_extra = catalog_dir + 'agnswide.dat'
file_sfg_extra = catalog_dir + 'sfgswide3.dat'

# Output
if dont_sample_flux:
	name_str = '_trecs'
else:
	name_str = '_%d' % i
catalog = catalog_dir + 'catalog' + name_str + '.dat'
print('Writing the catalog:', catalog)

file_allfluxes = npzs_dir + 'allfluxes' + name_str + '.npz'
file_fluxes = npzs_dir + 'fluxes' + name_str + '.npz'
file_histograms = npzs_dir + 'histograms' + name_str + '.npz'
file_base_fn = npzs_dir + 'base_fn' + name_str + '.npz'
####################################################################################################
# Create a new WCS object.  The number of axes must be set from the start
w = wcs.WCS(naxis=2)
# Set up a "Slant orthigraphic" projection
# Vector properties may be set with Python lists, or Numpy arrays
w.wcs.crpix = [crpix, crpix]
w.wcs.cdelt = np.array([cdeltx, cdelty])
w.wcs.crval = [center_ra, center_dec]
w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
w.wcs.set_pv([(2, 1, 0), (2, 2, 0)])


#################################################################################
#################################################################################
#################################################################################
"""
Sampling function (Andre)
"""
print("Do random fn =", random_fn)


class Flux_fn:
	"""
    Class to initialize sub threshld flux function and sample from it to generate training data.
    """
	def __init__(self, alpha=0, S0=1E-4,  crazy_test=False, random_fn=False):

		self.alpha = alpha
		self.S0 = S0
		self.SRange = np.linspace(np.log10(flux_min), np.log10(flux_max))
		self.sr_ = (25 * nat.degree ** 2).convert(nat.sr).value # 25deg^2
		self.crazy_test = crazy_test
		
		# ---------------------------------------------------------------------------------------------
		# Use Elisa's version N(s)
		base_name = 'histograms_flux_min_1e8_flux_max_1e0_ra_0_dec_0_Nsquares=1_nu_940MHz.npz'
		histo = np.load('npzs/%s' % (base_name)) # N(s) for the full T-RECS range
		bins = histo['bins_edges'] # bin edges
		widths = -(bins[:-1] - bins[1:])
		bins = (np.log10(bins[:-1]) + np.log10(bins[1:])) / 2 # bin centers
		hist = histo['histograms'][0]
		
		base_fn = interp1d(bins, np.log10((hist / widths))) # dN/ds in log log
		self.tot_range = bins
		
		# Crazy test function
		Gup = interp1d(self.SRange,
					   10 ** (np.log10((1 + (5E-5 / (10 ** (self.SRange)))) ** 0.29) + base_fn(self.SRange)))
		Gdown = interp1d(self.SRange,
						 10 ** (np.log10((1 + (5E-5 / (10 ** (self.SRange)))) ** (-0.5)) + base_fn(self.SRange)))

		rand_func = []
		for index, flux in enumerate(self.SRange):
			val = uniform(Gdown(flux), Gup(flux))
			rand_func.append(val)
		rand_fn = interp1d(self.SRange, rand_func)

		if self.crazy_test:
			self.base_fn = rand_fn
		
		elif change_functional_form:

			if  alt_functional_form == 1:
				n_s = interp1d(self.tot_range, 10 ** (
							np.log10( (1 + 0.1 * (5e-5 / 10 ** self.tot_range)**3 ) ** -0.4 ) + base_fn(self.tot_range)))
				self.N_s = interp1d(self.tot_range, n_s(self.tot_range) * widths)  # <--- Scale back to N(s)!!!!
				base_fn = interp1d(self.SRange, 10 ** (
							np.log10( (1 + 0.1 * (5e-5 / 10 ** self.SRange)**3 ) ** -0.4 ) + base_fn(self.SRange)))
				np.savez(file_base_fn, bins=self.SRange, base_fn=base_fn(self.SRange)) # this is dN/ds for the various parameter values
				self.base_fn = base_fn
				
			elif alt_functional_form == 2:
				n_s = interp1d(self.tot_range, 10 ** (
						np.log10( (1 - 0.2 * (0.7e-4/10**self.tot_range)**0.71 + 8 * (10e-7/10**self.tot_range)**1.75 ) ) + base_fn(self.tot_range)))
				self.N_s = interp1d(self.tot_range, n_s(self.tot_range) * widths)  # <--- Scale back to N(s)!!!!
				base_fn = interp1d(self.SRange, 10 ** (
						np.log10((1 - 0.2 * (0.7e-4/10**self.SRange)**0.71 + 8 * (10e-7/10**self.SRange)**1.75 )) + base_fn(self.SRange)))
				np.savez(file_base_fn, bins=self.SRange,
						 base_fn=base_fn(self.SRange))  # this is dN/ds for the various parameter values
				self.base_fn = base_fn
				
			elif alt_functional_form == 3:
				n_s = interp1d(self.tot_range, 10 ** (
						np.log10((1 + np.tanh(1e-5 / 10 ** self.tot_range)**4)) + base_fn(
					self.tot_range)))
				self.N_s = interp1d(self.tot_range, n_s(self.tot_range) * widths)  # <--- Scale back to N(s)!!!!
				base_fn = interp1d(self.SRange, 10 ** (
						np.log10((1 + np.tanh(1e-5 / 10 ** self.SRange)**4)) + base_fn(
					self.SRange)))
				np.savez(file_base_fn, bins=self.SRange,
						 base_fn=base_fn(self.SRange))  # this is dN/ds for the various parameter values
				self.base_fn = base_fn
			
		else:
			# dN/dS_Old = (1+S0/S)**alpha * dn/dS_New
			# base_fn   = interp1d(self.SRange,  2*self.SRange + np.log10((1 + (self.S0/(10**(self.SRange))))**alpha) + base_fn(self.SRange)    )
			n_s = interp1d(self.tot_range, 10 ** (
						np.log10((1 + self.S0 / 10 ** self.tot_range) ** alpha) + base_fn(self.tot_range)))
			self.N_s = interp1d(self.tot_range, n_s(self.tot_range) * widths)  # <--- Scale back to N(s)!!!!
			base_fn = interp1d(self.SRange, 10 ** (
						np.log10((1 + self.S0 / 10 ** self.SRange) ** alpha) + base_fn(self.SRange)))
			np.savez(file_base_fn, bins=self.SRange, base_fn=base_fn(self.SRange)) # this is dN/ds for the various parameter values
			self.base_fn = base_fn
		
		"""
        Sampling initialization
        - Calculate N_sources from 1E-7 up
        """
		
		def pN(logS):
			return self.base_fn(logS) * (10 ** (logS)) * np.log(10)

		
		def pFN(logS):
			return self.N_s(logS)
		
		self.p = interp1d(self.SRange, pFN(self.SRange))  # <--- Scale back to N(s)!!!!
		self.Tot_sources = quad(pN, min(self.SRange), max(self.SRange))[0]
		
		self.pmax = max(self.p(np.linspace(min(self.SRange), max(self.SRange))))
	
	def sample_full(self, N):
		Nbins = 12
		xmin = min(self.SRange)
		xmax = max(self.SRange)
		n_accept = 0
		x_list = []
		while n_accept < N:
			t = (xmax - xmin) * np.random.rand() + xmin
			y = np.random.rand()
			if y < self.p(t) / self.pmax:
				n_accept += 1
				x_list.append(t)
		x_list = np.array(x_list)
		hist, bins = np.histogram(x_list, bins=Nbins, density=True)
		width = 0.1
		center = (bins[:-1] + bins[1:]) / 2
		return center, hist * self.Tot_sources * (bins[1] - bins[0])
	
	def sample(self):
		"""
        Take a single sample
        """
		xmin = min(self.SRange)
		xmax = max(self.SRange)
		n_accept = 0
		x_list = -1
		while n_accept < 1:
			t = (xmax - xmin) * np.random.rand() + xmin
			y = np.random.rand()
			if y < self.p(t) / self.pmax:
				n_accept += 1
				x_list = t
		return x_list
	
	def sample_other_fn(self, fn, rnge):
		"""
        Take a single sample from interp1d object
        """
		fn_max = max(fn(rnge))
		xmin = min(rnge)
		xmax = max(rnge)
		n_accept = 0
		x_list = -1
		while n_accept < 1:
			t = (xmax - xmin) * np.random.rand() + xmin
			y = np.random.rand()
			if y < fn(t) / fn_max:
				n_accept += 1
				x_list = t
		return x_list
####################################################################################################
if  dont_sample_flux:
	N_sources =  int(1e10)
	N_agn     = int(np.round(frac_agn * N_sources))
	N_sfg     = N_sources - N_agn
	
else:
	if random_fn==False:
		# Set samping function
		G = Flux_fn(alpha=args.alpha[0], S0=args.S0[0], crazy_test=args.crazy_test)
	
	elif random_fn==True:
		class rd_Flux_fn():
			def __init__(self):
				l              = np.load('./npzs/histograms_2.npz')
				bins_edges           = l['bins_edges']
				
				alphas = [0, 0.2, -0.2, -0.2, -0.6, 0.3, 0.3, 0.21, 0.23, 0.23, -0.5, 0.29, 0.25, 0.27, -0.5, 0.27, 0.18, 0.13, 0.13, 0.17, -0.37]
				S0s = [0, 5e-05, 5e-06, 5e-05, 5e-05, 5e-06, 5e-05, 5e-05, 5e-06, 5e-05, 5e-05, 5e-05, 5e-05, 5e-05, 5e-06, 5e-06, 5e-06, 5e-06, 5e-05, 5e-05, 5e-05]
				rand_spec_list = []
				# Get bins in full range with same spacing as Elisa's
				edge_steps = (-2+7)/(np.log10(bins_edges[1])-np.log10(bins_edges[0])) + 1
				new_edges = np.linspace(np.log10(bins_edges[0]),-2, round(edge_steps))
				new_edges = 10**new_edges
				new_bins = (new_edges[:-1] + new_edges[1:]) / 2
				new_widths = -(new_edges[:-1] - new_edges[1:])
				new_bins = np.log10(new_bins)

				for alpha, S0 in zip(alphas, S0s):
						G          = Flux_fn(alpha=alpha,S0=S0)
						# Get points that have dumb shape but exist at preexisting training spots
						rand_spec_list.append(G.base_fn(new_bins))
				rand_spec_list = np.array(rand_spec_list)
				rand_spec_list = np.array([np.random.choice(x) for x in rand_spec_list.T])
				np.savez(file_base_fn,  bins=new_bins, base_fn=rand_spec_list)
				rand_spec_list = interp1d(new_bins,rand_spec_list)
				self.N_s = interp1d(new_bins, rand_spec_list(new_bins) * new_widths)  # <--- Scale back to N(s)!!!!
				self.new_bins = new_bins
				self.rand_spec_list = rand_spec_list
				
				def pN (logS):
					return rand_spec_list(logS) * (10**(logS)) * np.log(10)
				
				def pFN(logS):
					return self.N_s(logS)
				
				self.p = interp1d(self.new_bins, pFN(self.new_bins))
				self.pmax = max(self.p(self.new_bins))
				self.Tot_sources = quad(pN, min(new_bins), max(new_bins))[0]

				
			def sample(self):
				"""
				Take a single sample
				"""
				xmin  = min(self.new_bins)
				xmax  = max(self.new_bins)
				n_accept = 0
				x_list   = -1
				while n_accept < 1:
					t = (xmax-xmin)*np.random.rand() + xmin
					y = np.random.rand()
					if y < self.p(t)/ self.pmax:
						n_accept += 1
						x_list = t
				return x_list
			
		G         = rd_Flux_fn()
		
	N_sources =  np.round(G.Tot_sources)
	print("Total number of sources =" , N_sources)
	print("Total number of sources - lines in Trecs =" , (N_sources - N_trecsmedi))
	N_agn     = int(np.round(frac_agn * N_sources))
	N_sfg     = N_sources - N_agn
	print('N_agn, N_sfg =', N_agn, N_sfg)
	
#################################################################################
#################################################################################
#################################################################################
def large_ra(edges_ra):
	# converts negative ras into angles between 0 and 360 degrees
	j = 0
	for x in edges_ra:
		if x < 0:
			edges_ra[j] = x + 360
			j += 1
	
	return edges_ra


def large_ra_inv(x):
	# converts ras > 180 degrees into negative angles
	
	if x > 180:
		x = x - 360
	
	return x


def read_parameters(line, isAgn, isExtra):
	line = line.strip()
	line = " ".join(line.split())
	line = line.split(' ')
	
	# Read parameters
	if not isExtra:
		X = float(line[38]) # First angular coordinate for the flat-sky approximation
		Y = float(line[39]) # Second angular coordinate for the flat-sky approximation
	else:
		X, Y = np.random.uniform(-image_side / 2, image_side / 2, 2)
	
	flux780 = float(line[7]) / 1000  # flux at 780 MHz in Jy. T-Recs uses mJy
	flux1000 = float(line[8]) / 1000  # flux at 1 GHz in Jy. T-Recs uses mJy
	
	# Spectrum
	alpha = np.log(flux780 / flux1000) / lognunu0  # spectral index
	beta = 0  # spectral curvature, set to zero for small frequency range
	
	if dont_sample_flux:
		fluxnu = flux780 * (nu / .78) ** alpha
	else:
		# Sample function generates a sample of log10(S) from the PMF N(s).
		fluxnu = 10**G.sample()
	
	if isAgn:
		
		Rs = float(line[46])  # Ratio between the distance between the spots and the total size of the jets, for the FR I /FR II classification
		size = float(line[45]) / 3600  # Projected apparent size of the core+jet emission in degs. T-Recs uses arcsecs
		pop = line[47]  # Number identifying the sub-population: 4, 5, 6 for FSRQ, BL Lac and SS-AGNs, respectively.
		
		return X, Y, fluxnu, alpha, beta, Rs, size, pop
	
	else:
		
		e1 = float(line[44])  # First ellipticity component e1 = e sin(theta/2)
		e2 = float(line[45])  # Second ellipticity component e2 = e cos(theta/2)
		
		majorAxis = float(line[43]) / 3600  # major axis of the object in degrees. Corresponds to T-Recs "size", given in arcsec in T-Recs
		
		return X, Y, fluxnu, alpha, beta, e1, e2, majorAxis


def get_position(X, Y):
	# dec = center_dec + latitude  # declination in degrees. Corresponds to T-Recs "latitude"
	# thetam = dec * np.pi / 180
	# ra = center_ra + longitude / np.cos(thetam)  # right ascension in degrees. Corresponds to T-Recs "longitude"
	
	# Get pixel coordinates. Pixel counting starts from 1
	pixx = crpix + X/cdeltx
	pixy = crpix + Y/cdelty
	
	# Get corresponding celestial coordinates: ra, dec
	coords = w.wcs_pix2world([[pixx, pixy]], 1)
	ra = coords[0,0]
	dec = coords[0,1]
	return ra,dec, np.round(pixx), np.round(pixy)


def all_fluxes_from_exisiting_catalog(filename_catalog, icat):
	
	print('Reading catalog ', filename_catalog)
	pixy = []
	pixx = []
	fluxes = []  # array of fluxes in Jy
	
	with open(filename_catalog, "r") as fin:
		for i, line in enumerate(fin):
			# if i % 10000 ==0:
			# 	print(i, end=' ')
			if i==0:
				continue
				
			line = line.strip()
			line = " ".join(line.split())
			line = line.split(' ')
			# print(line)
			ra = float(line[0])
			dec = float(line[1])
			flux = float(line[2])
			
			pix = w.wcs_world2pix([[ra, dec]], 1)
			pixxx = np.round(pix[0, 0])
			pixxy = np.round(pix[0, 1])
			# print(ra, dec, pixxx, pixxy)
			pixy.append(pixxx)
			pixx.append(pixxy)
			fluxes.append(flux)
	
	np.savez(npzs_dir + 'allfluxes_%d.npz' %icat, fluxes=np.array(fluxes), pixy=np.array(pixy), pixx=np.array(pixx))


def clean_agn(filename, pixy, pixx, fluxes, isExtra, i_ini):
	print("\nGetting AGNs")
	
	i_agn = i_ini  # number of agns
	i_entries = 0  # number of entries in our catalogue. Differs from i_agn because Pop 6 agns are modeled as double sources.
	i_pointlike = 0  # number of pointlike sources
	i_bright = 0
	i_out = 0
	
	if not isExtra:
		openstring = "w"
	else:
		openstring = "a"
	
	with open(filename, "r") as fin:

		with open(catalog, openstring) as fout:
			
			if not isExtra:
				fout.write(header_str)
			
			for line in fin:
				
				X, Y, fluxnu, alpha, beta, Rs, size, pop = read_parameters(line, True, isExtra)
				if fluxnu < flux_min or fluxnu > flux_max:
					continue
					
				# Position
				ra, dec, pixxx, pixxy = get_position(X, Y)
				
				# Discard source if it is out of the image
				if pixxy < i_first_pixel or pixxx < i_first_pixel or pixxx > Npixel - 1 + i_first_pixel or pixxy > Npixel - 1 + i_first_pixel:
					i_out += 1
					continue

				if fluxnu >= flux_max_label:
					i_bright += 1
				# Size
				maxsize = size * Rs  # largest dimension of the object in degrees. Corresponds to T-Recs "size"*"Rs", i.e. projected distance between the two bright spots, given in arcsec in T-Recs
				
				double_source = False
				if pop == '4' or pop == '5':  # we model these sources as circles
					
					majorAxis = maxsize / 2  # major axis of the object in degrees. Corresponds to T-Recs "size"*"Rs"/2, i.e. half of the projected distance between the two bright spots, given in arcsec in T-Recs
					posAngle = 0
					
					if maxsize < resolution_limit:  # if the source is smaller than our resolution, we model it as pointlike
						majorAxis = 0
						i_pointlike += 1
					
					minorAxis = majorAxis  # minor axis of the object in degrees
					
					list1 = ['%.6f' % ra, '%.6f' % dec, '%.4e' % fluxnu, '%.6f' % alpha, str(beta), '%.6e' % majorAxis,
							 '%.6e' % minorAxis, str(posAngle)]
				
				else:
					
					if maxsize < resolution_limit:  # if the source is smaller than our resolution, we model it as pointlike
						
						i_pointlike += 1
						majorAxis = 0  # major axis of each object in degrees
						minorAxis = 0  # minor axis of each object in degrees
						posAngle = 0
						list1 = ['%.6f' % ra, '%.6f' % dec, '%.4e' % fluxnu, '%.6f' % alpha, str(beta),
								 '%.6e' % majorAxis, '%.6e' % minorAxis, str(posAngle)]
					
					else:
						
						double_source = True
						
						posAngle = np.random.uniform(0, 360)  # position angle in degrees
						
						# We model each of these sources (steep-spectum AGNs) as a pair of circular sources. Each source gets half the original flux
						# If Rs = 0, the two circular sources are exaclty on top of each other and the major axis is maxsize/2.
						# If Rs = 0.5, the two circular sources touch each other at one point and the major axis is maxsize/4.
						# For other value of Rs, we determine the major axis using a straight line interpolation between the two benchmarks above
						
						majorAxis = maxsize / 2 * (1. - Rs)  # major axis of each object in degrees
						minorAxis = majorAxis  # minor axis of each object in degrees
						
						ra1 = ra + maxsize / 2 * np.cos(
							posAngle)  # right ascension in degrees. Corresponds to T-Recs "longitude"
						dec1 = dec + maxsize / 2 * np.sin(
							posAngle)  # declination in degrees. Corresponds to T-Recs "latitude"
						
						ra2 = ra - maxsize / 2 * np.cos(
							posAngle)  # right ascension in degrees. Corresponds to T-Recs "longitude"
						dec2 = dec - maxsize / 2 * np.sin(
							posAngle)  # declination in degrees. Corresponds to T-Recs "latitude"
						
						list1 = ['%.6f' % ra1, '%.6f' % dec1, '%.4e' % (fluxnu / 2), '%.6f' % alpha, str(beta),
								 '%.6e' % majorAxis, '%.6e' % minorAxis, str(0)]  # each source gets half the total flux
						list2 = ['%.6f' % ra2, '%.6f' % dec2, '%.4e' % (fluxnu / 2), '%.6f' % alpha, str(beta),
								 '%.6e' % majorAxis, '%.6e' % minorAxis, str(0)]  # each source gets half the total flux
				
				s = "\t\t".join(list1) + "\n"
				fout.write(s)
				
				if double_source:
					i_entries += 1
					s = "\t\t".join(list2) + "\n"
					fout.write(s)
				
				pixy.append(pixxx)
				pixx.append(pixxy)
				fluxes.append(fluxnu)
				
				if i_agn >= N_agn - 1:
					i_agn += 1
					i_entries += 1
					print('Reached maximum # of AGNs. Exiting.', i_agn)
					print('# sources outside = ', i_out)
					return i_agn, i_entries, i_pointlike, i_bright
				i_agn += 1
				i_entries += 1
				
	print('# sources outside = ',i_out)
	return i_agn, i_entries, i_pointlike, i_bright


def clean_sfg(filename, pixy, pixx, fluxes, isExtra, i_ini):
	print("\nGetting SFGs")
	
	i_sfg = i_ini
	i_entries = 0  # number of entries in our catalogue. Differs from i_agn because Pop 6 agns are modeled as double sources.
	i_pointlike = 0  # number of pointlike sources
	i_out = 0
	i_bright = 0
	
	with open(filename, "r") as fin:
		# print('Reading file ', filename)
		with open(catalog, "a") as fout:
			
			for line in fin:
				
				X, Y, fluxnu, alpha, beta, e1, e2, majorAxis = read_parameters(line, False, isExtra)
				if fluxnu < flux_min or fluxnu > flux_max:
					continue
					
				# Position
				ra, dec, pixxx, pixxy = get_position(X, Y)
				
				# Discard source if it is out of the image
				if pixxy < i_first_pixel or pixxx< i_first_pixel or pixxx > Npixel - 1 + i_first_pixel or pixxy > Npixel - 1 + i_first_pixel:
					i_out += 1
					continue
				
				if fluxnu >= flux_max_label:
					i_bright += 1
					
				# print(i_sfg, N_sfg)
				isLast = (i_sfg == N_sfg - 1)
				
				if isLast:
					print('last line')

				# Size
				if majorAxis < resolution_limit:  # if the source is smaller than our resolution, we model it as pointlike
					majorAxis = 0
					posAngle = 0
					minorAxis = 0
					i_pointlike += 1

				else:
					posAngle = 2 * np.arctan2(e1, e2)  # position angle
					e = np.sqrt(e1 ** 2 + e2 ** 2)
					minorAxis = majorAxis * np.sqrt(1 - e**2)  # minor axis
				
				list1 = ['%.6f' % ra, '%.6f' % dec, '%.4e' % fluxnu, '%.6f' % alpha, str(beta), '%.6e' % majorAxis,
						 '%.6e' % minorAxis, str(posAngle)]
				
				# Write catalog
				if not isLast:
					s = "\t\t".join(list1) + "\n"
					fout.write(s)
				
				else:
					print("Writing last line")
					
					s = "\t\t".join(list1)
					fout.write(s)
				
				pixy.append(pixxx)
				pixx.append(pixxy)
				fluxes.append(fluxnu)
				
				if i_sfg >= N_sfg - 1:
					i_sfg += 1
					i_entries += 1
					print('Reached maximum # of SFGs. Exiting.', i_sfg)
					print('# sources outside = ', i_out)
					return i_sfg, i_entries, i_pointlike,  i_bright
				
				i_sfg += 1
				i_entries += 1
				
	print('# sources outside = ',i_out)
	return i_sfg, i_entries, i_pointlike, i_bright


def create_npz_and_catalog():
	
	pixy = []
	pixx = []
	fluxes = []  # array of fluxes in Jy
	
	i_agn_extra = 0
	i_entries_agn_extra = 0
	i_pointlike_agn_extra = 0
	i_bright_agn_extra = 0
	i_sfg_extra = 0
	i_entries_sfg_extra = 0
	i_pointlike_sfg_extra = 0
	i_bright_sfg_extra = 0
	
	i_agn, i_entries_agn, i_pointlike_agn, i_bright_agn = clean_agn(file_agn, pixy, pixx, fluxes, False, 0)
	print("# AGN = ", i_agn)
	print("# pointlike AGN = ", i_pointlike_agn)
	print("# bright AGN (> 1e-5 Jy) = ", i_bright_agn)
	print("# entries for AGN = ", i_entries_agn)


	if i_agn < N_agn:
		print("\nNeed some extra sources!!")
		i_agn_extra, i_entries_agn_extra, i_pointlike_agn_extra, i_bright_agn_extra = clean_agn(file_agn_extra, pixy, pixx, fluxes, True, i_agn)
		print("# AGN = ", i_agn_extra)
		print("# pointlike AGN = ", i_pointlike_agn + i_pointlike_agn_extra)
		print("# bright AGN (> 1e-5 Jy) = ", i_bright_agn + i_bright_agn_extra)
		print("# entries for AGN = ", i_entries_agn + i_entries_agn_extra)
	
	i_sfg, i_entries_sfg, i_pointlike_sfg, i_bright_sfg = clean_sfg(file_sfg, pixy, pixx, fluxes, False, 0)
	print("# SFG = ", i_sfg)
	print("# pointlike SFG = ", i_pointlike_sfg)
	print("# bright SFG (> 1e-5 Jy) = ", i_bright_sfg)
	print("# entries for SFG = ", i_entries_sfg)

	
	if i_sfg < N_sfg:
		print("\nNeed some extra sources!!")
		i_sfg_extra, i_entries_sfg_extra, i_pointlike_sfg_extra, i_bright_sfg_extra = clean_sfg(file_sfg_extra, pixy, pixx, fluxes, True, i_sfg)
	
		print("# SFG = ", i_sfg_extra)
		print("# pointlike SFG = ", i_pointlike_sfg + i_pointlike_sfg_extra)
		print("# bright SFG (> 1e-5 Jy) = ", i_bright_sfg + i_bright_sfg_extra)
		print("# entries for SFG = ", i_entries_sfg + i_entries_sfg_extra)
		
	np.savez(file_allfluxes, fluxes=np.array(fluxes), pixy=np.array(pixy), pixx=np.array(pixx))

	if i_agn_extra != 0 and i_sfg_extra != 0:
		i_tot = i_sfg_extra + i_agn_extra
	elif i_agn_extra != 0:
		i_tot =  i_sfg + i_agn_extra
	elif i_sfg_extra != 0:
		i_tot = i_sfg_extra + i_agn
	else:
		i_tot = i_sfg + i_agn
	
	print("\nDone.\n# of sources =", i_tot)
	print("# of pointlike =", i_pointlike_sfg_extra+i_pointlike_sfg +i_pointlike_agn_extra + i_pointlike_agn)
	print("# of entries =", i_entries_sfg + i_entries_agn + i_entries_sfg_extra + i_entries_agn_extra)
	print("# of bright of sources =", i_bright_sfg + i_bright_agn + i_bright_sfg_extra + i_bright_agn_extra)
	print("# of dim of sources =", i_tot - (i_bright_sfg + i_bright_agn + i_bright_sfg_extra + i_bright_agn_extra))


def create_npz_and_catalog_no_sampling():
	pixy = []
	pixx = []
	fluxes = []  # array of fluxes in Jy
	
	i_agn, i_entries_agn, i_pointlike_agn, i_bright_agn = clean_agn(file_agn, pixy, pixx, fluxes, False, 0)
	print("# AGN = ", i_agn)
	print("# pointlike AGN = ", i_pointlike_agn)
	print("# bright AGN (> 1e-4 Jy) = ", i_bright_agn)
	print("# entries for AGN = ", i_entries_agn)
	
	i_sfg, i_entries_sfg, i_pointlike_sfg, i_bright_sfg = clean_sfg(file_sfg, pixy, pixx, fluxes, False, 0)
	print("# SFG = ", i_sfg)
	print("# pointlike SFG = ", i_pointlike_sfg)
	print("# bright SFG (> 1e-4 Jy) = ", i_bright_sfg)
	print("# entries for SFG = ", i_entries_sfg)
	
	np.savez(file_allfluxes, fluxes=np.array(fluxes), pixy=np.array(pixy), pixx=np.array(pixx))
	
	i_tot = i_sfg + i_agn
	
	print("\nDone.\n# of sources =", i_tot)
	print("# of pointlike =",  i_pointlike_sfg  + i_pointlike_agn)
	print("# of entries =", i_entries_sfg + i_entries_agn )
	print("# of bright of sources =", i_bright_sfg + i_bright_agn )
	print("# of dim of sources =", i_tot - (i_bright_sfg + i_bright_agn))


def get_edges():
	# Edges of the squares into which the image is divided
	edges_pixx = np.linspace(i_first_pixel, Npixel + 1, num=Nstripes, endpoint=False)
	edges_pixy = np.linspace(i_first_pixel, Npixel + 1, num=Nstripes, endpoint=False)
	# print(edges_pixx.astype(int))
	# print(edges_pixy.astype(int))
	
	return edges_pixx, edges_pixy


def get_square_number(idxs, Nstripes):
	# the number of the top left square is 0
	return (idxs[1] - 1) * Nstripes + idxs[0] - 1


def split_into_squares():
	
	print('Splitting image into %d squares' % Nsquares)
	data = np.load(file_allfluxes)
	
	pixy = data['pixy']
	pixx = data['pixx']
	allfluxes = data['fluxes']
	
	fluxes = []

	for n in np.arange(Nsquares):
		fluxes.append([])

	edges_pixx, edges_pixy = get_edges()
	# print('Bin edges are ', edges_pixx.astype(int))


	for i in np.arange(len(pixy)):
		# numpy.digitize(x, bins, right=False)
		# Return the indices of the bins to which each value in input array belongs.
		# Returns: indices: ndarray of ints. Output: array of indices, of same shape as x.
		arr = [np.digitize(pixx[i], edges_pixx), np.digitize(pixy[i], edges_pixy)]
		squareN = get_square_number(arr, Nstripes)
		# print(pixx[i], pixy[i], arr, squareN)
		fluxes[squareN].append(allfluxes[i])

	# for i in np.arange(len(fluxes)):
	#     print(fluxes[i])
	print('Saving flux file')
	np.savez(file_fluxes, fluxes=np.array(fluxes, dtype=object))


def get_dNds():
	
	print('Creating histogram')
	fluxes = np.load(file_fluxes, allow_pickle=True)['fluxes']
	
	bins_edges = np.logspace(np.log10(flux_min_label), np.log10(flux_max_label), num=Nbins + 1,
							 endpoint=True)  # bin egdes
	# print(bins_edges)
	histograms = np.empty([len(fluxes), Nbins])
	
	i = 0
	for f in fluxes:
		histograms[i] = np.histogram(np.array(f), bins=bins_edges, density=False)[0]
		# print(histograms[i])
		i += 1
	
	np.savez(file_histograms, histograms=histograms, bins_edges=bins_edges)
	print(np.sum(histograms))


def create_dNds():
	split_into_squares()
	get_dNds()

def plot_dNds():
	
	for i in np.arange(9):
		data = np.load( npzs_dir + 'histograms_' + str(i) + '.npz')
		hist = data['histograms']
		bin_edges = data['bins_edges']
		bins =  (bin_edges[:-1] + bin_edges[1:]) / 2
		plt.plot(bins,hist[0], label=str(i) )
		
	plt.legend()
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(r'$s$')
	plt.ylabel(r'$N(s)$')
	plt.show()

def print_beam_pointing_coords():
	for X in [-1.5, -0.5, 0.5, 1.5]:
		for Y in [-1.5, -0.5, 0.5, 1.5]:
			ra, dec, pixx, pixy = get_position(X, Y)
			# print(ra, dec, pixx, pixy)
			c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
			print('%02dh%02dm%02.2f, %02d.%02d.%05.2f' % (
			c.ra.hms.h, c.ra.hms.m, c.ra.hms.s, c.dec.dms.d, np.abs(c.dec.dms.m), np.abs(c.dec.dms.s)))
			# print()
		
		
################################################################
################################################################
###############################################################
timer.tic()
if dont_sample_flux:
	create_npz_and_catalog_no_sampling()
else:
	create_npz_and_catalog()
timer.toc()
print()

timer.tic()
create_dNds()
timer.toc()





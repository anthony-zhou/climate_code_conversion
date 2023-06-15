from enum import Enum

class LeafRespMethod(Enum):
    RYAN1991 = 1
    ATKIN2015 = 2

class StomatalCondMethod(Enum):
    MEDLYN2011 = 1
    BALLBERRY1987 = 2

Input	value	range	units	definition	LH confidence
ci	40	[300, 800]	Pa	intracellular leaf CO2 (Pa)	high
fval	40	[300, 800]	Pa	ci from previous time step	medium
p, iv, c	1		index	These are array indexing terms 	high
gb_mol	500	[100, 1000]	(umol H2O/m2/s)	leaf boundary layer conductance (umol H2O/m2/s)	low 
je	40	[20,150]	(umol electrons/m2/s)	electron transport rate	high
cair	40	[35,60]	Pa	atmospheric CO2 partial pressure	high
oair	21000		Pa	atmospheric O2 partial pressure	high (sea level)
lmr_z	6	[0.1, 10]	(umol CO2/m2/s)	leaf maintenance respiration rate	medium (shouldn't this be a percentage?) not sure if this is leaf or canopy level,
par_z	500	[100,1500]	W/m2	Photosynthetically active radiation absorbed per unit lai for the canopy layer	medium - depends on canopy layer and LAI
rh_can	40	[0,100]	%	relative humidity in the canopy	high
gs_mol	10000	[1000,50000]	(umol H2O/m2/s)	leaf stomatal conductance	medium
atm2lnd_inst				This is an array of forcing data	
photosyns_inst				This is an array of plant parameters	
					
					
atm2lnd_inst%forc_pbot_downscaled_col	121000				
pftcon%medlynslope	6				
pftcon%medlynintercept	100				



def ci_func(ci,
            fval,
            p,
            iv,
            c,
            gb_mol,
            je,
            cair,
            oair,
            lmr_z,
            par_z,
            rh_can,
            atm2land_inst,
            photosyns_inst):
    # Assume we're using c3
    # Rubisco-limited photosynthesis (ac)
    # RuBP-limited photosynthesis (aj)
    # Product-limited photosynthesis (ap)

    # Co-limit ac and aj
    
 

# def photosynthesis(
#         bounds,
#         fn,
#         filterp,
#         esat_tv, 
#         eair,
#         oair
# )
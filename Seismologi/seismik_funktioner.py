from obspy.io.seg2.seg2 import SEG2
import obspy
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from inspect import signature

def læs_hammer_data(filnavne):
    """
    Læser hammerseismik data fra en array af filer
    
    Parametre:
    ----------
    filnavne : array 
        array af filnavnene fra hammerseismik forsøget
        
    Returnerer:
    ----------
    tider, amplituder, afstande
    hvor
    tider: array
        array over tidspunkterne data er taget efter hammerslag
    amplituder: 3-d array
        1. akse er tilsvarer hvert forsøg, hver fra en fil
        2. akse tilsvarer de 16 geofoner
        3. akse er hver af tiderne
    
    Eksempel:
    ----------
    filnavne = ["58.dat", "60.dat"]
    tider, amplituder, afstande = seismik_funktioner.læs_hammer_data(filnavne)
    amplituder[1,3,10] # "amplitude i 1. forsøg af 3. geofon til 10. tidspunkt"
    """
    if type(filnavne) is not list:
        print("Argumentet til \"read_campus_data\" skal være en array af filnavne. arrayn kan godt indeholde 1 fil, fx read_campus_data([\"100.dat\"])")
        return
    if (invalid := np.where([not os.path.exists(x) for x in filnavne])[0]).any():
        print(f"Argumentet til \"read_campus_data\" skal være en array af filnavne, men det er {np.array(filnavne)[invalid]} ikke. Hvis fil(erne) er i en underfolder, skal det være en del af filnavnet. Den nuværende folder er {os.getcwd()}/")
        return
    amplitudes = np.array([[x.data for x in SEG2().read_file(filename)] for filename in filnavne])
    times = SEG2().read_file(filnavne[0])[0].times()
    afstande = np.linspace(-1, 29, 16, endpoint=True)
    return times, amplitudes, afstande

def læs_marine_data(filnavn):
    """
    Læser marinedata fra fil
    
    Parametre:
    ----------
    filnavn : string 
        navn på marine data filen
        
    Returnerer:
    ----------
    tider, amplituder, afstande
    hvor
    tider: Array
        Array over tidspunkterne data er taget efter luftkanon
    amplituder: 3-d Array
        1. akse er tilsvarer hvert forsøg, hver fra en fil
        2. akse tilsvarer de mange hydrofoner
        3. akse er hver af tiderne
    
    Eksempel:
    ----------
    tider, amplituder, afstande = seismik_funktioner.læs_marine_data("kat15_line02_sel_raw.sgy")
    amplituder[1,3,10] # "amplitude i 1. forsøg af 3. hydrofon til 10. tidspunkt"
    """
    file = obspy.read(filnavn)
    amplitudes = np.array([x.data for x in file]).reshape(-1,48,2401)
    times = file[0].times()
    afstande = 6.4*np.arange(48) + 20
    return times, amplitudes, afstande

def plot_seismogram(tider, amplituder, afstande, skalaer = np.array([1]), funktioner=[]):  
    """
    Funktion til at visualisere seismometer data som en funktion af tid, efter den ønskede pre-processering
    
    Parametre:
    ----------
    tider : array 
        Array af tiderne data er blevet taget, i sekunder. 
    amplituder : 2d array
        En 2d-array med amplituder der skal plottes. 
        Hver række (1. dimension) repræsenterer en mikrofon, mens hver kolonne repræsenterer et tidspunkt
    afstande : array 
        Position af hver mikrofon, i meter. 
    skalaer : array
        Størrelse af svingninger i plottet for hver mikrofon.
    funktioner : array af funktioner
        funktioner der definerer linjer at tegne i plottet, defineret som tid(afstand).
    
    Eksempel:
    ----------
    def overflade(x): return x/10
    def overflade2(x): return x/20 + 0.1
    # Plot det første (altså [0]) forsøg, med skalaen justeret efter afstanden, sammen med 2 linjer med hældning hhv 1/10 og 1/20.
    seismik_funktioner.plot_seismogram(times, amplitudes[0], afstande, 
                                     skalaer=10*(np.abs(afstande)), 
                                     funktioner=[overflade,overflade2])
    """
    
    tider = np.array(tider)
    afstande = np.array(afstande)
    amplituder = np.array(amplituder)
    assert amplituder.ndim == 2, f"amplituder skal være en 2-dimensionel array, men du har givet et {amplituder.ndim} dimensionelt array. Har du måske givet data fra alle forsøg samlet? så kan du få den første med indexering: amplitudes[0]"
    assert tider.ndim == 1 and afstande.ndim == 1 and skalaer.ndim == 1, f"\"tider\", \"afstande\" og \"skalaer\" skal være 1-dimensionelle, men har her formerne hhv {tider.shape}, {afstande.shape} og {skalaer.shape}"
    assert len(tider) == amplituder.shape[1], f"\"amplituder\" har {amplituder.shape[1]} kolonner og derfor tidspunkter, men \"tider\" specificerer {len(tider)} tider."
    assert len(afstande) == amplituder.shape[0], f"\"amplituder\" har {amplituder.shape[0]} rækker hvilket skal være antallet af mikrofoner, men \"afstande\" specificerer afstanden til {len(afstande)} mikrofoner." 
    if callable(funktioner):
        funktioner = [funktioner]
    assert all([len(signature(f).parameters) == 1 for f in funktioner]), "argumentet \"funktioner\" skal bestå af funktioner T(x) som for enhver position x giver tiden til den position."
    
    N = len(afstande)
    dists = np.diff(afstande).mean()
    scaling = dists / np.nanmax(np.abs(amplituder)) 
    corr_amp = amplituder * skalaer[:,None]
    plt.figure(figsize=(10,7))
    for idx in range(N):
        xvalues = afstande[idx] + np.clip(corr_amp[idx] * scaling, -dists*0.9, dists*0.9)
        plt.fill_betweenx(tider, xvalues, np.minimum(xvalues, afstande[idx]), linestyle="-", color="k", linewidth=0.3, alpha=0.5)
        #plt.plot(afstande[idx] + (1*tider * corr_amp), tider,linewidth=0.7)
        plt.xlabel("distance [m]")
        plt.ylabel("time [s]")
        
    x_range = np.linspace(afstande[0]-dists, afstande[-1]+dists)
    for f in funktioner:
        plt.plot(x_range, list(map(f, x_range)), color="black", linestyle=(0, (5, 5)))

    plt.xlim(afstande[0]-dists,afstande[-1]+dists)
    plt.ylim(np.max(tider), np.min(tider))
   # plt.gca().invert_yaxis()


def højpass_filter(tider, amplituder, cutoff):
    """
    Funktion til at lave højpass filtrering af data for at fjerne lavfrekvent støj (fx bølger)
    
    Parametre:
    ----------
    tider : array 
        array af tiderne data er blevet taget, i sekunder. 
    amplituder : 2d array
        En 2d-array med amplituder der skal højpass filtreres
        Hver række (1. dimension) repræsenterer en mikrofon, mens hver kolonne repræsenterer et tidspunkt
    cutoff : kommatal
        periode over hvilken signal bliver fjernet, ie vi er kun intrasseret i bølger der svinger hurtigere end 'cutoff'
        
    Eksempel:
    ----------
        data = seismik_funktioner.højpass(times, amplitudes[0], 0.2)
    """
    normal_cutoff = (tider[1]-tider[0]) / (cutoff*0.5)
    # Get the filter coefficients 
    b, a = butter(2, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, amplituder)
    return y

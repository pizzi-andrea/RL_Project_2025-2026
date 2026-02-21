import numpy as np
from dataclasses import dataclass
from collections import OrderedDict

@dataclass(frozen=True)
class DoomNames:
    """Global player attribute names for standard VizDoom enviroment"""
    att_player_names: tuple[str]  = ('AMMO0', 'AMMO2','AMMOSELECTED_WEAPON', 'ARMOR', 'HEALTH', 'KILLCOUNT', 'SELECTED_WEAPON_AMMO')
    buttons_names: tuple[str]     =    ('NO-OPP', 'ATTACK', 'CROUCH', 'JUMP', 'MOVE_BACKWARD', 'MOVE_FORWARD', 'MOVE_LEFT', 'MOVE_RIGHT', 'RELOAD', 'SELECT_NEXT_WEAPON', 'SELECT_PREV_WEAPON', 'SPEED', 'STRAFE', 'TURN_LEFT', 'TURN_RIGHT', 'USE', 'ZOOM', "SELECT_WEAPON1", "SELECT_WEAPON2", "SELECT_WEAPON3", "SELECT_WEAPON4", "SELECT_WEAPON5", "SELECT_WEAPON6", "SELECT_WEAPON7")

@dataclass(frozen=True)
class DoomMap:
    """Global class indexs for standard VizDoom enviroment"""


    @staticmethod
    def local2global(values:list[int|float]|np.ndarray[int|float], names:list[str], global_order:list[str], one_hot:bool=True) -> dict[str, int]:
        
        d = OrderedDict.fromkeys(global_order, 0)
        v = values
        
        if one_hot:
            v = [0]*len(global_order)
            v[values] = 1

        for i, n in enumerate(names):
            d[n] = v[i]
        
        return d
    
    @staticmethod
    def reindex_categorical(name:str, global_order:list[str]) -> int:
        return global_order.index(name)
    
    @staticmethod
    def one_hot(v:int, n_classes:int) -> np.ndarray:
        one_hot = np.zeros(shape=n_classes, dtype=np.int8)
        one_hot[v] = 1
        return one_hot
    
    
    
    @staticmethod
    def global2local(values:list[int|float]|np.ndarray[int|float], names:list[str], global_order:list[str]) -> dict[str, int]:
        """
        Estrae i valori dei bottoni locali da un vettore globale.
        
        Args:
            values: Lista o array dei valori in ordine globale (len = len(global_order))
            names: Lista dei nomi dei bottoni locali (sottoinsieme locale)
            global_order: Lista dei nomi dei bottoni globali (riferimento)
            one_hot: Flag di simmetria (non strettamente necessario per l'estrazione, ma utile per assert logici)
            
        Returns:
            OrderedDict con chiavi = names e valori estratti dal vettore globale.
        """
        d = OrderedDict.fromkeys(names, 0)
        
        # Se viene passato un dizionario invece di una lista, estraiamo i valori se necessario,
        # ma la firma richiede list/ndarray. Assumiamo 'values' sia allineato a 'global_order'.
        
        for local_name in names:
            # Troviamo l'indice che questo bottone ha nel vettore globale
            if local_name in global_order:
                g_idx = global_order.index(local_name)
                # Assegniamo al dizionario locale il valore presente nel vettore globale
                d[local_name] = values[g_idx]
            else:
                # Gestione sicurezza: se un nome locale non esiste nel globale
                # (potrebbe capitare se la config è disallineata)
                pass 

        return d
    
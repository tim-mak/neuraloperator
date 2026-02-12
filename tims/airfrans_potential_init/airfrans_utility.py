import sys
import os

# Add the arbifoil directory to Python path
sys.path.insert(0, '/home/timm/Projects/PIML/arbifoil')
import arbifoil  


def  convert_airfrans_to_arbifoil(airfrans_data):
    """
    Convert AirFRANS dataset format to Arbifoil format.
    airfrans_data: Dict with keys 'x', 'y', 'props'
    Returns: Dict with keys 'input', 'output', 'properties'
    """
    arbifoil_data = {
        'input': airfrans_data['x'],  # Assuming this is the input tensor
        'output': airfrans_data['y'],  # Assuming this is the output tensor
        'properties': airfrans_data['props']  # Additional properties
    }
    return arbifoil_data

def calculate_inviscid_flow(coords,aoa):
    """
    Calculate inviscid flow properties using Arbifoil for given coordinates and angle of attack.
    coords: Tensor of shape [N, 2] with (x, y) coordinates
    aoa: Angle of attack in degrees
    Returns: Dict with keys 'u', 'v', 'p' representing velocity components and pressure
    """
    # Convert AoA to radians
    aoa_rad = torch.deg2rad(torch.tensor(aoa))
    
    # Create an Arbifoil instance and compute flow properties
    arbifoil_instance = arbifoil.Arbifoil(coords, aoa_rad)
    flow_properties = arbifoil_instance.compute_flow()
    
    return flow_properties
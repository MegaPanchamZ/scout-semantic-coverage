"""Reject malformed converted geometry before it can crash the simulator."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def audit_map(path: Path):
    root=ET.parse(path).getroot();issues=[]
    roads=root.findall('road')
    if root.tag!='OpenDRIVE' or not roads:issues.append('no_opendrive_roads')
    for road in roads:
        prefix='road:'+road.get('id','?')
        for g in road.findall('planView/geometry'):
            length=float(g.attrib['length'])
            if not math.isfinite(length) or length<=0:
                issues.append(prefix+':nonpositive_geometry_length')
        for element in road.iter():
            for name,value in element.attrib.items():
                try:numeric=float(value)
                except ValueError:continue
                if not math.isfinite(numeric):issues.append(prefix+':nonfinite_'+element.tag+'.'+name)
    return {'passed':not issues,'issues':issues,'road_count':len(roads)}

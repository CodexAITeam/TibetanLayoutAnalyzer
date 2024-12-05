import xml.etree.ElementTree as ET
from collections import defaultdict
from sickle import Sickle

def get_records(url = "https://oai.sbb.berlin/", set = "tibetica"):
  sickle = Sickle(url)
  records = sickle.ListRecords(metadataPrefix='mets', set=set)
  return records

def _strip_namespace(tag):
    """Remove namespace from the tag."""
    return tag.split('}', 1)[1] if '}' in tag else tag


def _element_to_dict(element, remove_namespace):
    """
    Recursively convert an XML element and its children into a dictionary.
    """
    parsed_data = defaultdict(list)

    # Include attributes as keys
    for key, value in element.attrib.items():
        parsed_data[_strip_namespace(key) if remove_namespace else key] = value

    # Parse child elements
    for child in element:
        tag = _strip_namespace(child.tag) if remove_namespace else child.tag
        parsed_data[tag].append(_element_to_dict(child, remove_namespace))

    # Include text if present
    if element.text and element.text.strip():
        parsed_data["text"] = element.text.strip()

    return dict(parsed_data)


def parse_stabi_metadata(stabi_record, remove_namespace=True):
    """
    Parse an XML string into a nested dictionary, optionally removing namespaces.

    Parameters:
        stabi_record (str): OAI record.
        remove_namespace (bool): If True, namespaces will be stripped from the keys.

    Returns:
        dict: A nested dictionary representation of the XML data.
    """

    xml_string = stabi_record.raw

    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Convert the XML root element to a dictionary
    root_tag = _strip_namespace(root.tag) if remove_namespace else root.tag

    return _element_to_dict(root, remove_namespace)
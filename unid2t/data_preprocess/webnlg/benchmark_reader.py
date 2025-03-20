import xml.etree.ElementTree as Et
from collections import defaultdict
from os import listdir
import json
from xml.dom import minidom
import copy


class Triple:

    def __init__(self, s, p, o):
        self.s = s
        self.o = o
        self.p = p

    def flat_triple(self):
        return self.s + ' | ' + self.p + ' | ' + self.o


class Tripleset:

    def __init__(self):
        self.triples = []
        self.clusterid = 0

    def fill_tripleset(self, t):
        for xml_triple in t:
            s, p, o = xml_triple.text.split(' | ')
            triple = Triple(s, p, o)
            self.triples.append(triple)


class Lexicalisation:

    def __init__(self, lex, lid, comment='', lang=''):
        self.lex = lex
        self.id = lid
        self.comment = comment
        self.lang = lang

    def chars_length(self):
        return len(self.lex)


class Entry:

    def __init__(self, category, size, eid, shape, shape_type):
        self.category = category
        self.size = size
        self.id = eid
        self.shape = shape
        self.shape_type = shape_type
        self.originaltripleset = []
        self.modifiedtripleset = Tripleset()
        self.lexs = []
        self.dbpedialinks = []
        self.links = []

    def fill_originaltriple(self, xml_t):
        otripleset = Tripleset()
        self.originaltripleset.append(otripleset)   # multiple originaltriplesets for one entry
        otripleset.fill_tripleset(xml_t)

    def fill_modifiedtriple(self, xml_t):
        self.modifiedtripleset.fill_tripleset(xml_t)

    def create_lex(self, xml_lex):
        try:
            comment = xml_lex.attrib['comment']
        except KeyError:
            comment = ''
        try:
            lang = xml_lex.attrib['lang']
        except KeyError:
            lang = ''
        lid = xml_lex.attrib['lid']
        lex = Lexicalisation(xml_lex.text, lid, comment, lang)
        self.lexs.append(lex)

    def create_dbpedialinks(self, xml_dbpedialinks):
        for xml_dblink in xml_dbpedialinks:
            s, p, o = xml_dblink.text.split(' | ')
            dbp_link = Triple(s, p, o)
            self.dbpedialinks.append(dbp_link)

    def create_links(self, xml_links):
        for xml_link in xml_links:
            s, p, o = xml_link.text.split(' | ')
            link = Triple(s, p, o)
            self.links.append(link)

    def count_lexs(self):
        return len(self.lexs)

    def flat_tripleset(self):
        """
        Render modified triples to the flat representation with <br>.
        :return: flat representation
        """
        flat_mr = []
        for triple in self.modifiedtripleset.triples:
            flat_triple = triple.s + ' | ' + triple.p + ' | ' + triple.o
            flat_mr.append(flat_triple)
        if self.size == '1':
            return flat_mr[0]
        else:
            return '<br>'.join(flat_mr)

    def relations(self):
        """
        Give a set of properties found in tripleset
        :return: set of properties
        """
        rel_set = set()
        for triple in self.modifiedtripleset.triples:
            rel_set.add(triple.p)
        return rel_set

    def list_triples(self):
        """
        Return a list of triples for an entry.
        :return: list of triples
        """
        triples = []
        for triple in self.modifiedtripleset.triples:
            flat_triple = triple.s + ' | ' + triple.p + ' | ' + triple.o
            triples.append(flat_triple)
        return triples


class Benchmark:

    def __init__(self):
        self.entries = []

    def fill_benchmark(self, fileslist):
        """
        Parse xml files and fill Benchmark with Entry instances.
        :param fileslist: [(path_to_file, filename.xml), (), ... ()]
        :return:
        """
        for file in fileslist:
            myfile = file[0] + '/' + file[1]
            tree = Et.parse(myfile)
            root = tree.getroot()
            for xml_entry in root.iter('entry'):
                entry_id = xml_entry.attrib['eid']
                category = xml_entry.attrib['category']
                size = xml_entry.attrib['size']
                shape = xml_entry.attrib['shape']
                shape_type = xml_entry.attrib['shape_type']

                entry = Entry(category, size, entry_id, shape, shape_type)
                for child in xml_entry:
                    if child.tag == 'originaltripleset':
                        entry.fill_originaltriple(child)
                    elif child.tag == 'modifiedtripleset':
                        entry.fill_modifiedtriple(child)
                    elif child.tag == 'lex':
                        entry.create_lex(child)
                    elif child.tag == 'dbpedialinks':
                        entry.create_dbpedialinks(child)
                    elif child.tag == 'links':
                        entry.create_links(child)
                self.entries.append(entry)

    def total_lexcount(self):
        count = [entry.count_lexs() for entry in self.entries]
        return sum(count)

    def unique_p_otriples(self):
        properties = [triple.p for entry in self.entries for triple in entry.originaltripleset[0].triples]
        return set(properties)

    def unique_p_mtriples(self):
        properties = [triple.p for entry in self.entries for triple in entry.modifiedtripleset.triples]
        return set(properties)

    def entry_count(self, size=None, cat=None):
        """
        calculate the number of entries in benchmark
        :param size: size (should be string)
        :param cat: category
        :return: entry count
        """
        if not size and cat:
            entries = [entry for entry in self.entries if entry.category == cat]
        elif not cat and size:
            entries = [entry for entry in self.entries if entry.size == size]
        elif not size and not cat:
            return len(self.entries)
        else:
            entries = [entry for entry in self.entries if entry.category == cat and entry.size == size]
        return len(entries)

    def lexcount_size_category(self, size, cat):
        """Calculate the number of lexicalisations."""
        counts = [entry.count_lexs() for entry in self.entries if entry.category == cat and entry.size == size]
        return sum(counts)

    def property_map(self):
        """Approximate mapping between modified properties and original properties.
        Don't extract all mappings!"""
        mprop_oprop = defaultdict(set)
        for entry in self.entries:
            for tripleset in entry.originaltripleset:
                for i, triple in enumerate(tripleset.triples):
                    m_property = entry.modifiedtripleset.triples[i].p
                    m_subj = entry.modifiedtripleset.triples[i].s
                    m_obj = entry.modifiedtripleset.triples[i].o
                    if m_subj == triple.s and m_obj == triple.o:  # we are losing some mappings here
                        mprop_oprop[m_property].add(triple.p)
                    if not mprop_oprop[m_property]:  # some false positives can stem from here
                        mprop_oprop[m_property].add(triple.p)
        return mprop_oprop

    def filter(self, size=[], cat=[]):
        """
        Filter set of entries in the benchmark wrt size and category.
        :param size: list of triple sizes (str) to extract; default empty -- all sizes
        :param cat: list of categories to extract; default empty -- all categories
        :return: copied benchmark object with filtered size and categories;
                if no entry is left, return None
        """
        bench_filtered = self.copy()
        for entry in self.entries:
            deleted = False
            if cat:
                if entry.category not in cat:
                    bench_filtered.del_entry(entry)
                    deleted = True
            if size and not deleted:
                if entry.size not in size:
                    bench_filtered.del_entry(entry)
        if bench_filtered.entries:
            return bench_filtered
        else:
            return None

    def copy(self):
        """
        Copy all the benchmark objects.
        :return: a deep copy list
        """
        b_copy = Benchmark()
        b_copy.entries = copy.deepcopy(self.entries)
        return b_copy

    def filter_by_entry_ids(self, entry_ids):
        """
        Filter corpus and leave only entries whose ids are present in 'entry_ids'.
        :param entry_ids: a list of entry ids
        :return: a new bmk object with filtered entries
        """
        bench_filtered = self.copy()
        for entry in self.entries:
            if entry.id not in entry_ids:
                bench_filtered.del_entry(entry)
        return bench_filtered

    def triplesets(self):
        """
        List of all modified triplesets.
        :return: a list of objects Tripleset
        """
        all_triplesets = [entry.modifiedtripleset for entry in self.entries]
        return all_triplesets

    def del_entry(self, entry):
        """Delete an entry from Benchmark.
        The condition on ids is useful when we want to delete an entry from the Benchmark copy
        using the Benchmark original ids."""
        for init_entry in self.entries:
            if init_entry.id == entry.id:
                self.entries.remove(init_entry)

    def get_lex_by_id(self, entry_category, entry_size, entry_id, lex_id):
        """Get lexicalisation by supplying entry and lex ids."""
        for entry in self.entries:
            if entry.id == entry_id and entry.size == entry_size and entry.category == entry_category:
                for lex in entry.lexs:
                    if lex.id == lex_id:
                        return lex.lex

    def subjects_objects(self):
        subjects = set()
        objects = set()
        for entry in self.entries:
            for triple in entry.originaltripleset[0].triples:
                subjects.add(triple.s)
                objects.add(triple.o)
        return subjects, objects

    def verbalisations(self):
        """Get all lexicalisations."""
        verbalisations = []
        for entry in self.entries:
            for lex in entry.lexs:
                verbalisations.append(lex.lex)
        return verbalisations

    def sort_by_size_and_name(self):
        # sort by size, and then by flat mtripleset name
        sorted_entries = sorted(self.entries, key=lambda x: (x.size, x.flat_tripleset()))
        self.entries = sorted_entries
        return self

    def b2json(self, path, filename):
        """Convert benchmark to json."""
        data = {}
        data['entries'] = []
        entry_id = 0  # new entry ids
        for entry in self.entries:
            entry_id += 1
            orig_triplesets = {}
            orig_triplesets['originaltripleset'] = []
            modif_tripleset = []
            lexs = []
            links = []
            dbpedialinks = []
            for otripleset in entry.originaltripleset:
                orig_tripleset = []
                for triple in otripleset.triples:
                    orig_tripleset.append({'subject': triple.s, 'property': triple.p, 'object': triple.o})
                orig_triplesets['originaltripleset'].append(orig_tripleset)

            for triple in entry.modifiedtripleset.triples:
                modif_tripleset.append({'subject': triple.s, 'property': triple.p, 'object': triple.o})

            for lex in entry.lexs:
                lexs.append({'comment': lex.comment, 'xml_id': lex.id, 'lex': lex.lex, 'lang': lex.lang})

            if entry.dbpedialinks:
                for link in entry.dbpedialinks:
                    dbpedialinks.append({'subject': link.s, 'property': link.p, 'object': link.o})

            if entry.links:
                for link in entry.links:
                    links.append({'subject': link.s, 'property': link.p, 'object': link.o})

            data['entries'].append({entry_id: {'category': entry.category, 'size': entry.size, 'xml_id': entry.id,
                                               'shape': entry.shape, 'shape_type': entry.shape_type,
                                               'originaltriplesets': orig_triplesets,
                                               'modifiedtripleset': modif_tripleset,
                                               'lexicalisations': lexs,
                                               'dbpedialinks': dbpedialinks,
                                               'links': links}
                                    })

        with open(path + '/' + filename, 'w+', encoding='utf8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4, sort_keys=True)

    def b2xml(self, path, filename, recalc_id=True):
        """Convert benchmark to pretty xml."""
        root = Et.Element('benchmark')
        entries_xml = Et.SubElement(root, 'entries')
        for index, entry in enumerate(self.entries):
            if recalc_id:
                entry.id = str(index + 1)
            entry_xml = Et.SubElement(entries_xml, 'entry',
                                      attrib={'category': entry.category, 'eid': entry.id, 'size': entry.size,
                                              'shape': entry.shape, 'shape_type': entry.shape_type})
            for otripleset in entry.originaltripleset:
                otripleset_xml = Et.SubElement(entry_xml, 'originaltripleset')
                for triple in otripleset.triples:
                    otriple_xml = Et.SubElement(otripleset_xml, 'otriple')
                    otriple_xml.text = triple.s + ' | ' + triple.p + ' | ' + triple.o
            mtripleset_xml = Et.SubElement(entry_xml, 'modifiedtripleset')
            for mtriple in entry.modifiedtripleset.triples:
                mtriple_xml = Et.SubElement(mtripleset_xml, 'mtriple')
                mtriple_xml.text = mtriple.s + ' | ' + mtriple.p + ' | ' + mtriple.o
            for lex in entry.lexs:
                lex_xml = Et.SubElement(entry_xml, 'lex', attrib={'comment': lex.comment, 'lid': lex.id,
                                                                  'lang': lex.lang})
                lex_xml.text = lex.lex
            if entry.dbpedialinks:
                dbpedialinks_xml = Et.SubElement(entry_xml, 'dbpedialinks')
                for link in entry.dbpedialinks:
                    dbpedialink_xml = Et.SubElement(dbpedialinks_xml, 'dbpedialink', attrib={'direction': 'en2ru'})
                    dbpedialink_xml.text = link.s + ' | ' + link.p + ' | ' + link.o
            if entry.links:
                links_xml = Et.SubElement(entry_xml, 'links')
                for link in entry.links:
                    link_xml = Et.SubElement(links_xml, 'link', attrib={'direction': 'en2ru'})
                    link_xml.text = link.s + ' | ' + link.p + ' | ' + link.o

        ugly_xml_string = Et.tostring(root, encoding='utf-8', method='xml')
        xml = minidom.parseString(ugly_xml_string).toprettyxml(indent='  ')
        with open(path + '/' + filename, 'w+', encoding='utf-8') as f:
            f.write(xml)

    @staticmethod
    def categories():
        return ['Airport', 'Artist', 'Astronaut', 'Athlete', 'Building', 'CelestialBody', 'City',
                'ComicsCharacter', 'Company', 'Food', 'MeanOfTransportation', 'Monument',
                'Politician', 'SportsTeam', 'University', 'WrittenWork']


def select_files(topdir, category='', size=(1, 8)):
    finaldirs = [topdir+'/'+str(item)+'triples' for item in range(size[0], size[1])]

    finalfiles = []
    for item in finaldirs:
        finalfiles += [(item, filename) for filename in sorted(listdir(item)) if category in filename and '.xml' in filename]
    return finalfiles

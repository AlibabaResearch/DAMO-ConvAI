/*!*
 * Javascript BibTex Parser v0.1
 * Copyright (c) 2008 Simon Fraser University
 * @author Steve Hannah <shannah at sfu dot ca>
 *
 *
 * License:
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * Credits:
 *
 * This library is a port of the PEAR Structures_BibTex parser written
 * in PHP (http://pear.php.net/package/Structures_BibTex).
 *
 * In order to make porting the parser into javascript easier, I have made
 * use of many phpjs functions, which are distributed here under the MIT License:
 *
 *
 * More info at: http://kevin.vanzonneveld.net/techblog/category/php2js
 *
 * php.js is copyright 2008 Kevin van Zonneveld.
 *
 * Portions copyright Ates Goral (http://magnetiq.com), Legaev Andrey,
 * _argos, Jonas Raoni Soares Silva (http://www.jsfromhell.com),
 * Webtoolkit.info (http://www.webtoolkit.info/), Carlos R. L. Rodrigues, Ash
 * Searle (http://hexmen.com/blog/), Tyler Akins (http://rumkin.com), mdsjack
 * (http://www.mdsjack.bo.it), Alexander Ermolaev
 * (http://snippets.dzone.com/user/AlexanderErmolaev), Andrea Giammarchi
 * (http://webreflection.blogspot.com), Bayron Guevara, Cord, David, Karol
 * Kowalski, Leslie Hoare, Lincoln Ramsay, Mick@el, Nick Callen, Peter-Paul
 * Koch (http://www.quirksmode.org/js/beat.html), Philippe Baumann, Steve
 * Clay, booeyOH
 *
 * Licensed under the MIT (MIT-LICENSE.txt) license.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL KEVIN VAN ZONNEVELD BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * Synopsis:
 * ----------
 *
 * This class provides the following functionality:
 *    1. Parse BibTex into a logical data javascript data structure.
 *    2. Output parsed BibTex entries as HTML, RTF, or BibTex.
 *
 *
 * The following usage instructions have been copyed and adapted from the PHP instructions located
 * at http://pear.php.net/manual/en/package.structures.structures-bibtex.intro.php
 * Introduction
 * --------------
 * Overview
 * ----------
 * This package provides methods to access information stored in a BibTex
 * file. During parsing it is possible to let the data be validated. In
 * addition. the creation of BibTex Strings as well as RTF Strings is also
 * supported. A few examples
 *
 * Example 1. Loading a BibTex File and printing the parsed array
 * <script src="BibTex.js"></script>
 * <script>
 * bibtex = new BibTex();
 * bibtex.content = content; // the bibtex content as a string
 *
 * bibtex->parse();
 * alert(print_r($bibtex->data,true));
 * </script>
 *
 *
 * Options
 * --------
 * Options can be set either in the constructor or with the method
 * setOption(). When setting in the constructor the options are given in an
 * associative array. The options are:
 *
 * 	-	stripDelimiter (default: true) Stripping the delimiter surrounding the entries.
 * 	-	validate (default: true) Validation while parsing.
 * 	-	unwrap (default: false) Unwrapping entries while parsing.
 * 	-	wordWrapWidth (default: false) If set to a number higher one
 * 	    that the entries are wrapped after that amount of characters.
 * 	-	wordWrapBreak (default: \n) String used to break the line (attached to the line).
 * 	-	wordWrapCut (default: 0) If set to zero the line will we
 * 	    wrapped at the next possible space, if set to one the line will be
 * 	    wrapped exactly after the given amount of characters.
 * 	-	removeCurlyBraces (default: false) If set to true Curly Braces will be removed.
 *
 * Example of setting options in the constructor:
 *
 * Example 2. Setting options in the constructor
 * bibtex = new BibTex({'validate':false, 'unwrap':true});
 *
 *
 * Example of setting options using the method setOption():
 *
 * Example 62-3. Setting options using setOption
 * bibtex = new BibTex();
 * bibtex.setOption('validate', false);
 * bibtex.setOption('unwrap', true);
 *
 * Stored Data
 * ------------
 * The data is stored in the class variable data. This is a a list where
 * each entry is a hash table representing one bibtex-entry. The keys of
 * the hash table correspond to the keys used in bibtex and the values are
 * the corresponding values. Some of these keys are:
 *
 * 	-	cite - The key used in a LaTeX source to do the citing.
 * 	-	entryType - The type of the entry, like techreport, book and so on.
 * 	-	author - One or more authors of the entry. This entry is also a
 * 	    list with hash tables representing the authors as entries. The
 * 	    author has table is explained later.
 * 	-	title - Title of the entry.
 *
 * Author
 * ------
 * As described before the authors are stored in a list. Every entry
 * representing one author as a has table. The hash table consits of four
 * keys: first, von, last and jr. The keys are explained in the following
 * list:
 *
 * 	-	first - The first name of the author.
 * 	-	von - Some names have a 'von' part in their name. This is usually a sign of nobleness.
 * 	-	last - The last name of the author.
 * 	-	jr - Sometimes a author is the son of his father and has the
 * 	    same name, then the value would be jr. The same is true for the
 * 	    value sen but vice versa.
 *
 * Adding an entry
 * ----------------
 * To add an entry simply create a hash table with the needed keys and
 * values and call the method addEntry().
 * Example 4. Adding an entry
 * bibtex                         = new BibTex();
 * var addarray                   = {};
 * addarray['entryType']          = 'Article';
 * addarray['cite']               = 'art2';
 * addarray['title']              = 'Titel of the Article';
 * addarray['author'] = [];
 * addarray['author'][0]['first'] = 'John';
 * addarray['author'][0]['last']  = 'Doe';
 * addarray['author'][1]['first'] = 'Jane';
 * addarray['author'][1]['last']  = 'Doe';
 * bibtex.addEntry(addarray);
 */

// ------------BEGIN PHP FUNCTIONS -------------------------------------------------------------- //

// {{{ array
function array( ) {
    // #!#!#!#!# array::$descr1 does not contain valid 'array' at line 258
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_array/
    // +       version: 805.1716
    // +   original by: d3x
    // *     example 1: array('Kevin', 'van', 'Zonneveld');
    // *     returns 1: ['Kevin', 'van', 'Zonneveld'];

    return Array.prototype.slice.call(arguments);
}// }}}

// {{{ array_key_exists
function array_key_exists ( key, search ) {
    // Checks if the given key or index exists in the array
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_array_key_exists/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +   improved by: Felix Geisendoerfer (http://www.debuggable.com/felix)
    // *     example 1: array_key_exists('kevin', {'kevin': 'van Zonneveld'});
    // *     returns 1: true

    // input sanitation
    if( !search || (search.constructor !== Array && search.constructor !== Object) ){
        return false;
    }

    return key in search;
}// }}}// {{{ array_keys
function array_keys( input, search_value, strict ) {
    // Return all the keys of an array
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_array_keys/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: array_keys( {firstname: 'Kevin', surname: 'van Zonneveld'} );
    // *     returns 1: {0: 'firstname', 1: 'surname'}

    var tmp_arr = new Array(), strict = !!strict, include = true, cnt = 0;

    for ( key in input ){
        include = true;
        if ( search_value != undefined ) {
            if( strict && input[key] !== search_value ){
                include = false;
            } else if( input[key] != search_value ){
                include = false;
            }
        }

        if( include ) {
            tmp_arr[cnt] = key;
            cnt++;
        }
    }

    return tmp_arr;
}// }}}

// {{{ in_array
function in_array(needle, haystack, strict) {
    // Checks if a value exists in an array
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_in_array/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: in_array('van', ['Kevin', 'van', 'Zonneveld']);
    // *     returns 1: true

    var found = false, key, strict = !!strict;

    for (key in haystack) {
        if ((strict && haystack[key] === needle) || (!strict && haystack[key] == needle)) {
            found = true;
            break;
        }
    }

    return found;
}// }}}

// {{{ sizeof
function sizeof ( mixed_var, mode ) {
    // Alias of count()
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_sizeof/
    // +       version: 804.1712
    // +   original by: Philip Peterson
    // -    depends on: count
    // *     example 1: sizeof([[0,0],[0,-4]], 'COUNT_RECURSIVE');
    // *     returns 1: 6
    // *     example 2: sizeof({'one' : [1,2,3,4,5]}, 'COUNT_RECURSIVE');
    // *     returns 2: 6

    return count( mixed_var, mode );
}// }}}

// {{{ count
function count( mixed_var, mode ) {
    // Count elements in an array, or properties in an object
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_count/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +      input by: _argos
    // *     example 1: count([[0,0],[0,-4]], 'COUNT_RECURSIVE');
    // *     returns 1: 6
    // *     example 2: count({'one' : [1,2,3,4,5]}, 'COUNT_RECURSIVE');
    // *     returns 2: 6

    var key, cnt = 0;

    if( mode == 'COUNT_RECURSIVE' ) mode = 1;
    if( mode != 1 ) mode = 0;

    for (key in mixed_var){
        cnt++;
        if( mode==1 && mixed_var[key] && (mixed_var[key].constructor === Array || mixed_var[key].constructor === Object) ){
            cnt += count(mixed_var[key], 1);
        }
    }

    return cnt;
}// }}}

// {{{ explode
function explode( delimiter, string, limit ) {
    // Split a string by string
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_explode/
    // +       version: 805.1715
    // +     original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +     improved by: kenneth
    // +     improved by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +     improved by: d3x
    // +     bugfixed by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: explode(' ', 'Kevin van Zonneveld');
    // *     returns 1: {0: 'Kevin', 1: 'van', 2: 'Zonneveld'}
    // *     example 2: explode('=', 'a=bc=d', 2);
    // *     returns 2: ['a', 'bc=d']

    var emptyArray = { 0: '' };

    // third argument is not required
    if ( arguments.length < 2
        || typeof arguments[0] == 'undefined'
        || typeof arguments[1] == 'undefined' )
    {
        return null;
    }

    if ( delimiter === ''
        || delimiter === false
        || delimiter === null )
    {
        return false;
    }

    if ( typeof delimiter == 'function'
        || typeof delimiter == 'object'
        || typeof string == 'function'
        || typeof string == 'object' )
    {
        return emptyArray;
    }

    if ( delimiter === true ) {
        delimiter = '1';
    }

    if (!limit) {
        return string.toString().split(delimiter.toString());
    } else {
        // support for limit argument
        var splitted = string.toString().split(delimiter.toString());
        var partA = splitted.splice(0, limit - 1);
        var partB = splitted.join(delimiter.toString());
        partA.push(partB);
        return partA;
    }
}// }}}

// {{{ implode
function implode( glue, pieces ) {
    // Join array elements with a string
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_implode/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +   improved by: _argos
    // *     example 1: implode(' ', ['Kevin', 'van', 'Zonneveld']);
    // *     returns 1: 'Kevin van Zonneveld'

    return ( ( pieces instanceof Array ) ? pieces.join ( glue ) : pieces );
}// }}}

// {{{ join
function join( glue, pieces ) {
    // Alias of implode()
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_join/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // -    depends on: implode
    // *     example 1: join(' ', ['Kevin', 'van', 'Zonneveld']);
    // *     returns 1: 'Kevin van Zonneveld'

    return implode( glue, pieces );
}// }}}

// {{{ split
function split( delimiter, string ) {
    // Split string into array by regular expression
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_split/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // -    depends on: explode
    // *     example 1: split(' ', 'Kevin van Zonneveld');
    // *     returns 1: {0: 'Kevin', 1: 'van', 2: 'Zonneveld'}

    return explode( delimiter, string );
}// }}}

// {{{ str_replace
function str_replace(search, replace, subject) {
    // Replace all occurrences of the search string with the replacement string
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_str_replace/
    // +       version: 805.3114
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +   improved by: Gabriel Paderni
    // +   improved by: Philip Peterson
    // +   improved by: Simon Willison (http://simonwillison.net)
    // +    revised by: Jonas Raoni Soares Silva (http://www.jsfromhell.com)
    // -    depends on: is_array
    // *     example 1: str_replace(' ', '.', 'Kevin van Zonneveld');
    // *     returns 1: 'Kevin.van.Zonneveld'
    // *     example 2: str_replace(['{name}', 'l'], ['hello', 'm'], '{name}, lars');
    // *     returns 2: 'hemmo, mars'

    var f = search, r = replace, s = subject;
    var ra = is_array(r), sa = is_array(s), f = [].concat(f), r = [].concat(r), i = (s = [].concat(s)).length;

    while (j = 0, i--) {
        while (s[i] = s[i].split(f[j]).join(ra ? r[j] || "" : r[0]), ++j in f){};
    };

    return sa ? s : s[0];
}// }}}

// {{{ strlen
function strlen( string ){
    // Get string length
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_strlen/
    // +       version: 805.1616
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +   improved by: Sakimori
    // *     example 1: strlen('Kevin van Zonneveld');
    // *     returns 1: 19

    return ("" + string).length;
}// }}}

// {{{ strpos
function strpos( haystack, needle, offset){
    // Find position of first occurrence of a string
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_strpos/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: strpos('Kevin van Zonneveld', 'e', 5);
    // *     returns 1: 14

    var i = haystack.indexOf( needle, offset ); // returns -1
    return i >= 0 ? i : false;
}// }}}

// {{{ strrpos
function strrpos( haystack, needle, offset){
    // Find position of last occurrence of a char in a string
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_strrpos/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: strrpos('Kevin van Zonneveld', 'e');
    // *     returns 1: 16

    var i = haystack.lastIndexOf( needle, offset ); // returns -1
    return i >= 0 ? i : false;
}// }}}

// {{{ strtolower
function strtolower( str ) {
    // Make a string lowercase
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_strtolower/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: strtolower('Kevin van Zonneveld');
    // *     returns 1: 'kevin van zonneveld'

    return str.toLowerCase();
}// }}}

// {{{ strtoupper
function strtoupper( str ) {
    // Make a string uppercase
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_strtoupper/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: strtoupper('Kevin van Zonneveld');
    // *     returns 1: 'KEVIN VAN ZONNEVELD'

    return str.toUpperCase();
}// }}}

// {{{ substr
function substr( f_string, f_start, f_length ) {
    // Return part of a string
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_substr/
    // +       version: 804.1712
    // +     original by: Martijn Wieringa
    // *         example 1: substr('abcdef', 0, -1);
    // *         returns 1: 'abcde'

    if(f_start < 0) {
        f_start += f_string.length;
    }

    if(f_length == undefined) {
        f_length = f_string.length;
    } else if(f_length < 0){
        f_length += f_string.length;
    } else {
        f_length += f_start;
    }

    if(f_length < f_start) {
        f_length = f_start;
    }

    return f_string.substring(f_start, f_length);
}// }}}

// {{{ trim
function trim( str, charlist ) {
    // Strip whitespace (or other characters) from the beginning and end of a string
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_trim/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +   improved by: mdsjack (http://www.mdsjack.bo.it)
    // +   improved by: Alexander Ermolaev (http://snippets.dzone.com/user/AlexanderErmolaev)
    // +      input by: Erkekjetter
    // +   improved by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +      input by: DxGx
    // +   improved by: Steven Levithan (http://blog.stevenlevithan.com)
    // *     example 1: trim('    Kevin van Zonneveld    ');
    // *     returns 1: 'Kevin van Zonneveld'
    // *     example 2: trim('Hello World', 'Hdle');
    // *     returns 2: 'o Wor'
    if (!str) { return ''; }
    var whitespace;

    if(!charlist){
        whitespace = ' \n\r\t\f\x0b\xa0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u200b\u2028\u2029\u3000';
    } else{
        whitespace = charlist.replace(/([\[\]\(\)\.\?\/\*\{\}\+\$\^\:])/g, '\$1');
    }

	for (var i = 0; i < str.length; i++) {
		if (whitespace.indexOf(str.charAt(i)) === -1) {
		str = str.substring(i);
		break;
		}
	}
	for (i = str.length - 1; i >= 0; i--) {
		if (whitespace.indexOf(str.charAt(i)) === -1) {
			str = str.substring(0, i + 1);
			break;
    	}
	}
	return whitespace.indexOf(str.charAt(0)) === -1 ? str : '';
}// }}}


// {{{ wordwrap
function wordwrap( str, int_width, str_break, cut ) {
    // Wraps a string to a given number of characters
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_wordwrap/
    // +       version: 804.1715
    // +   original by: Jonas Raoni Soares Silva (http://www.jsfromhell.com)
    // +   improved by: Nick Callen
    // +    revised by: Jonas Raoni Soares Silva (http://www.jsfromhell.com)
    // *     example 1: wordwrap('Kevin van Zonneveld', 6, '|', true);
    // *     returns 1: 'Kevin |van |Zonnev|eld'

    var m = int_width, b = str_break, c = cut;
    var i, j, l, s, r;

    if(m < 1) {
        return str;
    }
    for(i = -1, l = (r = str.split("\n")).length; ++i < l; r[i] += s) {
        for(s = r[i], r[i] = ""; s.length > m; r[i] += s.slice(0, j) + ((s = s.slice(j)).length ? b : "")){
            j = c == 2 || (j = s.slice(0, m + 1).match(/\S*(\s)?$/))[1] ? m : j.input.length - j[0].length || c == 1 && m || j.input.length + (j = s.slice(m).match(/^\S*/)).input.length;
        }
    }

    return r.join("\n");
}// }}}

// {{{ is_string
function is_string( mixed_var ){
    // Find whether the type of a variable is string
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_is_string/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: is_string('23');
    // *     returns 1: true
    // *     example 2: is_string(23.5);
    // *     returns 2: false

    return (typeof( mixed_var ) == 'string');
}// }}}


// {{{ ord
function ord( string ) {
    // Return ASCII value of character
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_ord/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: ord('K');
    // *     returns 1: 75

    return string.charCodeAt(0);
}// }}}

// {{{ array_unique
function array_unique( array ) {
    // Removes duplicate values from an array
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_array_unique/
    // +       version: 805.211
    // +   original by: Carlos R. L. Rodrigues (http://www.jsfromhell.com)
    // +      input by: duncan
    // +    bufixed by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // *     example 1: array_unique(['Kevin','Kevin','van','Zonneveld']);
    // *     returns 1: ['Kevin','van','Zonneveld']

    var p, i, j, tmp_arr = array;
    for(i = tmp_arr.length; i;){
        for(p = --i; p > 0;){
            if(tmp_arr[i] === tmp_arr[--p]){
                for(j = p; --p && tmp_arr[i] === tmp_arr[p];);
                i -= tmp_arr.splice(p + 1, j - p).length;
            }
        }
    }

    return tmp_arr;
}// }}}

// {{{ print_r
function print_r( array, return_val ) {
    // Prints human-readable information about a variable
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_print_r/
    // +       version: 805.2023
    // +   original by: Michael White (http://crestidg.com)
    // +   improved by: Ben Bryan
    // *     example 1: print_r(1, true);
    // *     returns 1: 1

    var output = "", pad_char = " ", pad_val = 4;

    var formatArray = function (obj, cur_depth, pad_val, pad_char) {
        if (cur_depth > 0) {
            cur_depth++;
        }

        var base_pad = repeat_char(pad_val*cur_depth, pad_char);
        var thick_pad = repeat_char(pad_val*(cur_depth+1), pad_char);
        var str = "";

        if (obj instanceof Array || obj instanceof Object) {
            str += "Array\n" + base_pad + "(\n";
            for (var key in obj) {
                if (obj[key] instanceof Array || obj[key] instanceof Object) {
                    str += thick_pad + "["+key+"] => "+formatArray(obj[key], cur_depth+1, pad_val, pad_char);
                } else {
                    str += thick_pad + "["+key+"] => " + obj[key] + "\n";
                }
            }
            str += base_pad + ")\n";
        } else {
            str = obj.toString();
        }

        return str;
    };

    var repeat_char = function (len, pad_char) {
        var str = "";
        for(var i=0; i < len; i++) {
            str += pad_char;
        };
        return str;
    };
    output = formatArray(array, 0, pad_val, pad_char);

    if (return_val !== true) {
        document.write("<pre>" + output + "</pre>");
        return true;
    } else {
        return output;
    }
}// }}}
// {{{ is_array
function is_array( mixed_var ) {
    // Finds whether a variable is an array
    //
    // +    discuss at: http://kevin.vanzonneveld.net/techblog/article/javascript_equivalent_for_phps_is_array/
    // +       version: 804.1712
    // +   original by: Kevin van Zonneveld (http://kevin.vanzonneveld.net)
    // +   improved by: Legaev Andrey
    // +   bugfixed by: Cord
    // *     example 1: is_array(['Kevin', 'van', 'Zonneveld']);
    // *     returns 1: true
    // *     example 2: is_array('Kevin van Zonneveld');
    // *     returns 2: false

    return ( mixed_var instanceof Array );
}// }}}

//------------END PHP FUNCTIONS --------------------------------------------------------------   //

/**
 * BibTex
 *
 * A class which provides common methods to access and
 * create Strings in BibTex format+
 * Example 1: Parsing a BibTex File and returning the number of entries
 * <code>
 * bibtex = new BibTex();
 * bibtex.content = '....';
 *
 * bibtex.parse();
 * print "There are "+bibtex.amount()+" entries";
 * </code>
 * Example 2: Parsing a BibTex File and getting all Titles
 * <code>
 * bibtex = new BibTex();
 * bibtex.content="...";
 * bibtex.parse();
 * for (var i in bibtex.data) {
 *  alert( bibtex.data[i]['title']+"<br />");
 * }
 * </code>
 * Example 3: Adding an entry and printing it in BibTex Format
 * <code>
 * bibtex                         = new BibTex();
 * addarray                       = {}
 * addarray['entryType']          = 'Article';
 * addarray['cite']               = 'art2';
 * addarray['title']              = 'Titel2';
 * addarray['author']			  = [];
 * addarray['author'][0]['first'] = 'John';
 * addarray['author'][0]['last']  = 'Doe';
 * addarray['author'][1]['first'] = 'Jane';
 * addarray['author'][1]['last']  = 'Doe';
 * bibtex.addEntry(addarray);
 * alert( nl2br(bibtex.bibTex()));
 * </code>
 *
 * @category   Structures
 * @package    BibTex
 * @author     Steve Hannah <shannah at sfu dot ca>
 * @adapted-from Structures_BibTex by  Elmar Pitschke <elmar+pitschke@gmx+de>
 * @copyright  2008 Simon Fraser University
 * @license    http://www.gnu.org/licenses/lgpl.html
 * @version    Release: 0.1
 * @link       http://webgroup.fas.sfu.ca/projects/JSBibTexParser
 */
function BibTex(options)
{

	if ( typeof options == 'undefined' ) options = {};
    /**
     * Array with the BibTex Data
     *
     * @access public
     * @var array
     */
    this.data;
    /**
     * String with the BibTex content
     *
     * @access public
     * @var string
     */
    this.content;
    /**
     * Array with possible Delimiters for the entries
     *
     * @access private
     * @var array
     */
    this._delimiters;
    /**
     * Array to store warnings
     *
     * @access public
     * @var array
     */
    this.warnings;
    /**
     * Run-time configuration options
     *
     * @access private
     * @var array
     */
    this._options;
    /**
     * RTF Format String
     *
     * @access public
     * @var string
     */
    this.rtfstring;
    /**
     * HTML Format String
     *
     * @access public
     * @var string
     */
    this.htmlstring;
    /**
     * Array with the "allowed" entry types
     *
     * @access public
     * @var array
     */
    this.allowedEntryTypes;
    /**
     * Author Format Strings
     *
     * @access public
     * @var string
     */
    this.authorstring;

    this._delimiters     = {'"':'"',
                                        '{':'}'};
	this.data            = [];
	this.content         = '';
	//this._stripDelimiter = stripDel;
	//this._validate       = val;
	this.warnings        = [];
	this._options        = {
		'stripDelimiter'    : true,
		'validate'          : true,
		'unwrap'            : false,
		'wordWrapWidth'     : false,
		'wordWrapBreak'     : "\n",
		'wordWrapCut'       : 0,
		'removeCurlyBraces' : false,
		'extractAuthors'    : true
	};
	for (option in options) {
		test = this.setOption(option, options[option]);
		if (this.isError(test)) {
			//Currently nothing is done here, but it could for example raise an warning
		}
	}
	this.rtfstring         = 'AUTHORS, "{\b TITLE}", {\i JOURNAL}, YEAR';
	this.htmlstring        = 'AUTHORS, "<strong>TITLE</strong>", <em>JOURNAL</em>, YEAR<br />';
	this.allowedEntryTypes = array(
		'article',
		'book',
		'booklet',
		'confernce',
		'inbook',
		'incollection',
		'inproceedings',
		'manual',
		'masterthesis',
		'misc',
		'phdthesis',
		'proceedings',
		'techreport',
		'unpublished'
	);
	this.authorstring = 'VON LAST, JR, FIRST';

}


BibTex.prototype = {

    /**
     * Constructor
     *
     * @access public
     * @return void
     */


    /**
     * Sets run-time configuration options
     *
     * @access public
     * @param string option option name
     * @param mixed  value value for the option
     * @return mixed true on success PEAR_Error on failure
     */
    setOption : function(option, value)
    {
        ret = true;
        if (array_key_exists(option, this._options)) {
            this._options[option] = value;
        } else {
            ret = this.raiseError('Unknown option '+option);
        }
        return ret;
    },

    /**
     * Reads a give BibTex File
     *
     * @access public
     * @param string filename Name of the file
     * @return mixed true on success PEAR_Error on failure
     *
    function loadFile(filename)
    {
        if (file_exists(filename)) {
            if ((this.content = @file_get_contents(filename)) === false) {
                return PEAR::raiseError('Could not open file '+filename);
            } else {
                this._pos    = 0;
                this._oldpos = 0;
                return true;
            }
        } else {
            return PEAR::raiseError('Could not find file '+filename);
        }
    }
	*/
    /**
     * Parses what is stored in content and clears the content if the parsing is successfull+
     *
     * @access public
     * @return boolean true on success and PEAR_Error if there was a problem
     */
    parse: function()
    {
    	//alert("starting to parse");
        //The amount of opening braces is compared to the amount of closing braces
        //Braces inside comments are ignored
        this.warnings = [];
        this.data     = [];
        var valid          = true;
        var open           = 0;
        var entry          = false;
        var charv           = '';
        var lastchar       = '';
        var buffer         = '';
        for (var i = 0; i < strlen(this.content); i++) {
            charv = substr(this.content, i, 1);
            if ((0 != open) && ('@' == charv)) {
                if (!this._checkAt(buffer)) {
                    this._generateWarning('WARNING_MISSING_END_BRACE', '', buffer);
                    //To correct the data we need to insert a closing brace
                    charv     = '}';
                    i--;
                }
            }
            if ((0 == open) && ('@' == charv)) { //The beginning of an entry
                entry = true;
            } else if (entry && ('{' == charv) && ('\\' != lastchar)) { //Inside an entry and non quoted brace is opening
                open++;
            } else if (entry && ('}' == charv) && ('\\' != lastchar)) { //Inside an entry and non quoted brace is closing
                open--;
                if (open < 0) { //More are closed than opened
                    valid = false;
                }
                if (0 == open) { //End of entry
                    entry     = false;
                    var entrydata = this._parseEntry(buffer);
                    if (!entrydata) {
                        /**
                         * This is not yet used+
                         * We are here if the Entry is either not correct or not supported+
                         * But this should already generate a warning+
                         * Therefore it should not be necessary to do anything here
                         */
                    } else {
                        this.data[this.data.length] = entrydata;
                    }
                    buffer = '';
                }
            }
            if (entry) { //Inside entry
                buffer += charv;
            }
            lastchar = charv;
        }
        //If open is one it may be possible that the last ending brace is missing
        if (1 == open) {
            entrydata = this._parseEntry(buffer);
            if (!entrydata) {
                valid = false;
            } else {
                this.data[this.data.length] = entrydata;
                buffer = '';
                open   = 0;
            }
        }
        //At this point the open should be zero
        if (0 != open) {
            valid = false;
        }
        //Are there Multiple entries with the same cite?
        if (this._options['validate']) {
            cites = [];
            for (var i=0; i< this.data.length; i++ ) {
                cites[cites.length] = this.data[i]['cite'];
            }
            unique = array_unique(cites);
            if (cites.length != sizeof(unique)) { //Some values have not been unique!
                notuniques = [];
                for (var i = 0; i < cites.length; i++) {
                    if ('' == unique[i]) {
                        notuniques[notuniques.length] = cites[i];
                    }
                }
                this._generateWarning('WARNING_MULTIPLE_ENTRIES', implode(',',notuniques));
            }
        }
        //alert("finished parsing");
        if (valid) {
            this.content = '';
            return true;
        } else {
            return this.raiseError('Unbalanced parenthesis');
        }
    },

    /**
     * Extracting the data of one content
     *
     * The parse function splits the content into its entries+
     * Then every entry is parsed by this function+
     * It parses the entry backwards+
     * First the last '=' is searched and the value extracted from that+
     * A copy is made of the entry if warnings should be generated+ This takes quite
     * some memory but it is needed to get good warnings+ If nor warnings are generated
     * then you don have to worry about memory+
     * Then the last ',' is searched and the field extracted from that+
     * Again the entry is shortened+
     * Finally after all field:value pairs the cite and type is extraced and the
     * authors are splitted+
     * If there is a problem false is returned+
     *
     * @access private
     * @param string entry The entry
     * @return array The representation of the entry or false if there is a problem
     */
    '_parseEntry': function(entry)
    {
        var entrycopy = '';
        if (this._options['validate']) {
            entrycopy = entry; //We need a copy for printing the warnings
        }
        var ret = {};
        if ('@string' ==  strtolower(substr(entry, 0, 7))) {
            //String are not yet supported!
            if (this._options['validate']) {
                this._generateWarning('STRING_ENTRY_NOT_YET_SUPPORTED', '', entry+'}');
            }
        } else if ('@preamble' ==  strtolower(substr(entry, 0, 9))) {
            //Preamble not yet supported!
            if (this._options['validate']) {
                this._generateWarning('PREAMBLE_ENTRY_NOT_YET_SUPPORTED', '', entry+'}');
            }
        } else {
            //Parsing all fields
            while (strrpos(entry,'=') !== false) {
                position = strrpos(entry, '=');
                //Checking that the equal sign is not quoted or is not inside a equation (For example in an abstract)
                proceed  = true;
                if (substr(entry, position-1, 1) == '\\') {
                    proceed = false;
                }
                if (proceed) {
                    proceed = this._checkEqualSign(entry, position);
                }
                while (!proceed) {
                    substring = substr(entry, 0, position);
                    position  = strrpos(substring,'=');
                    proceed   = true;
                    if (substr(entry, position-1, 1) == '\\') {
                        proceed = false;
                    }
                    if (proceed) {
                        proceed = this._checkEqualSign(entry, position);
                    }
                }

                value = trim(substr(entry, position+1));
                entry = substr(entry, 0, position);

                if (',' == substr(value, strlen(value)-1, 1)) {
                    value = substr(value, 0, -1);
                }
                if (this._options['validate']) {
                    this._validateValue(value, entrycopy);
                }
                if (this._options['stripDelimiter']) {
                    value = this._stripDelimiter(value);
                }
                if (this._options['unwrap']) {
                    value = this._unwrap(value);
                }
                if (this._options['removeCurlyBraces']) {
                    value = this._removeCurlyBraces(value);
                }
                position    = strrpos(entry, ',');
                field       = strtolower(trim(substr(entry, position+1)));
                ret[field] = value;
                entry       = substr(entry, 0, position);
            }
            //Parsing cite and entry type
            var arr = entry.split('{');
            ret['cite'] = trim(arr[1]);
            ret['entryType'] = strtolower(trim(arr[0]));
            //alert(array_keys(ret));
            if ('@' == ret['entryType'].substring(0,1)) {
                ret['entryType'] = substr(ret['entryType'], 1);
            }
            if (this._options['validate']) {
                if (!this._checkAllowedEntryType(ret['entryType'])) {
                    this._generateWarning('WARNING_NOT_ALLOWED_ENTRY_TYPE', ret['entryType'], entry+'}');
                }
            }
            //Handling the authors
            if (in_array('author', array_keys(ret)) && this._options['extractAuthors']) {
                ret['author'] = this._extractAuthors(ret['author']);
            }
        }
        return ret;
    },

    /**
     * Checking whether the position of the '=' is correct
     *
     * Sometimes there is a problem if a '=' is used inside an entry (for example abstract)+
     * This method checks if the '=' is outside braces then the '=' is correct and true is returned+
     * If the '=' is inside braces it contains to a equation and therefore false is returned+
     *
     * @access private
     * @param string entry The text of the whole remaining entry
     * @param int the current used place of the '='
     * @return bool true if the '=' is correct, false if it contains to an equation
     */
    '_checkEqualSign': function(entry, position)
    {
        var ret = true;
        //This is getting tricky
        //We check the string backwards until the position and count the closing an opening braces
        //If we reach the position the amount of opening and closing braces should be equal
        var length = strlen(entry);
        var open   = 0;
        for (var i = length-1; i >= position; i--) {
            precedingchar = substr(entry, i-1, 1);
            charv          = substr(entry, i, 1);
            if (('{' == charv) && ('\\' != precedingchar)) {
                open++;
            }
            if (('}' == charv) && ('\\' != precedingchar)) {
                open--;
            }
        }
        if (0 != open) {
            ret = false;
        }
        //There is still the posibility that the entry is delimited by double quotes+
        //Then it is possible that the braces are equal even if the '=' is in an equation+
        if (ret) {
            entrycopy = trim(entry);
            lastchar  = substr(entrycopy,strlen(entrycopy)-1,1);
            if (',' == lastchar) {
                lastchar = substr(entrycopy, strlen(entrycopy)-2, 1);
            }
            if ('"' == lastchar) {
                //The return value is set to false
                //If we find the closing " before the '=' it is set to true again+
                //Remember we begin to search the entry backwards so the " has to show up twice - ending and beginning delimiter
                ret = false;
                found = 0;
                for (var i = length; i >= position; i--) {
                    precedingchar = substr(entry, i-1, 1);
                    charv          = substr(entry, i, 1);
                    if (('"' == charv) && ('\\' != precedingchar)) {
                        found++;
                    }
                    if (2 == found) {
                        ret = true;
                        break;
                    }
                }
            }
        }
        return ret;
    },

    /**
     * Checking if the entry type is allowed
     *
     * @access private
     * @param string entry The entry to check
     * @return bool true if allowed, false otherwise
     */
    '_checkAllowedEntryType': function(entry)
    {
        return in_array(entry, this.allowedEntryTypes);
    },

    /**
     * Checking whether an at is outside an entry
     *
     * Sometimes an entry misses an entry brace+ Then the at of the next entry seems to be
     * inside an entry+ This is checked here+ When it is most likely that the at is an opening
     * at of the next entry this method returns true+
     *
     * @access private
     * @param string entry The text of the entry until the at
     * @return bool true if the at is correct, false if the at is likely to begin the next entry+
     */
    '_checkAt': function(entry)
    {
        var ret     = false;
        var opening = array_keys(this._delimiters);
        var closing = array_values(this._delimiters);
        //Getting the value (at is only allowd in values)
        if (strrpos(entry,'=') !== false) {
            position = strrpos(entry, '=');
            proceed  = true;
            if (substr(entry, position-1, 1) == '\\') {
                proceed = false;
            }
            while (!proceed) {
                substring = substr(entry, 0, position);
                position  = strrpos(substring,'=');
                proceed   = true;
                if (substr(entry, position-1, 1) == '\\') {
                    proceed = false;
                }
            }
            value    = trim(substr(entry, position+1));
            open     = 0;
            charv     = '';
            lastchar = '';
            for (var i = 0; i < strlen(value); i++) {
                charv = substr(this.content, i, 1);
                if (in_array(charv, opening) && ('\\' != lastchar)) {
                    open++;
                } else if (in_array(charv, closing) && ('\\' != lastchar)) {
                    open--;
                }
                lastchar = charv;
            }
            //if open is grater zero were are inside an entry
            if (open>0) {
                ret = true;
            }
        }
        return ret;
    },

    /**
     * Stripping Delimiter
     *
     * @access private
     * @param string entry The entry where the Delimiter should be stripped from
     * @return string Stripped entry
     */
    '_stripDelimiter': function(entry)
    {
        var beginningdels = array_keys(this._delimiters);
        var ength        = strlen(entry);
        var firstchar     = substr(entry, 0, 1);
        var lastchar      = substr(entry, -1, 1);
        while (in_array(firstchar, beginningdels)) { //The first character is an opening delimiter
            if (lastchar == this._delimiters[firstchar]) { //Matches to closing Delimiter
                entry = substr(entry, 1, -1);
            } else {
                break;
            }
            firstchar = substr(entry, 0, 1);
            lastchar  = substr(entry, -1, 1);
        }
        return entry;
    },

    /**
     * Unwrapping entry
     *
     * @access private
     * @param string entry The entry to unwrap
     * @return string unwrapped entry
     */
    '_unwrap': function(entry)
    {
        entry = entry.replace(/\s+/, ' ');
        return trim(entry);
    },

    /**
     * Wordwrap an entry
     *
     * @access private
     * @param string entry The entry to wrap
     * @return string wrapped entry
     */
    '_wordwrap': function(entry)
    {
        if ( (''!=entry) && (is_string(entry)) ) {
            entry = wordwrap(entry, this._options['wordWrapWidth'], this._options['wordWrapBreak'], this._options['wordWrapCut']);
        }
        return entry;
    },

    /**
     * Extracting the authors
     *
     * @access private
     * @param string entry The entry with the authors
     * @return array the extracted authors
     */
    '_extractAuthors': function(entry) {
        entry       = this._unwrap(entry);
        var authorarray = entry.split(' and ');
        for (var i = 0; i < authorarray.length; i++) {
            var author = trim(authorarray[i]);
            /*The first version of how an author could be written (First von Last)
             has no commas in it*/
            var first    = '';
            var von      = '';
            var last     = '';
            var jr       = '';
            if (strpos(author, ',') === false) {
                var tmparray = author.split(' |~');
                var size     = tmparray.length;
                if (1 == size) { //There is only a last
                    last = tmparray[0];
                } else if (2 == size) { //There is a first and a last
                    first = tmparray[0];
                    last  = tmparray[1];
                } else {
                    var invon  = false;
                    var inlast = false;
                    for (var j=0; j<(size-1); j++) {
                        if (inlast) {
                            last += ' '+tmparray[j];
                        } else if (invon) {
                            casev = this._determineCase(tmparray[j]);
                            if (this.isError(casev)) {
                                // IGNORE?
                            } else if ((0 == casev) || (-1 == casev)) { //Change from von to last
                                //You only change when there is no more lower case there
                                islast = true;
                                for (var k=(j+1); k<(size-1); k++) {
                                    futurecase = this._determineCase(tmparray[k]);
                                    if (this.isError(casev)) {
                                        // IGNORE?
                                    } else if (0 == futurecase) {
                                        islast = false;
                                    }
                                }
                                if (islast) {
                                    inlast = true;
                                    if (-1 == casev) { //Caseless belongs to the last
                                        last += ' '+tmparray[j];
                                    } else {
                                        von  += ' '+tmparray[j];
                                    }
                                } else {
                                    von    += ' '+tmparray[j];
                                }
                            } else {
                                von += ' '+tmparray[j];
                            }
                        } else {
                            var casev = this._determineCase(tmparray[j]);
                            if (this.isError(casev)) {
                                // IGNORE?
                            } else if (0 == casev) { //Change from first to von
                                invon = true;
                                von   += ' '+tmparray[j];
                            } else {
                                first += ' '+tmparray[j];
                            }
                        }
                    }
                    //The last entry is always the last!
                    last += ' '+tmparray[size-1];
                }
            } else { //Version 2 and 3
                var tmparray     = [];
                tmparray     = explode(',', author);
                //The first entry must contain von and last
                vonlastarray = [];
                vonlastarray = explode(' ', tmparray[0]);
                size         = sizeof(vonlastarray);
                if (1==size) { //Only one entry.got to be the last
                    last = vonlastarray[0];
                } else {
                    inlast = false;
                    for (var j=0; j<(size-1); j++) {
                        if (inlast) {
                            last += ' '+vonlastarray[j];
                        } else {
                            if (0 != (this._determineCase(vonlastarray[j]))) { //Change from von to last
                                islast = true;
                                for (var k=(j+1); k<(size-1); k++) {
                                    this._determineCase(vonlastarray[k]);
                                    casev = this._determineCase(vonlastarray[k]);
                                    if (this.isError(casev)) {
                                        // IGNORE?
                                    } else if (0 == casev) {
                                        islast = false;
                                    }
                                }
                                if (islast) {
                                    inlast = true;
                                    last   += ' '+vonlastarray[j];
                                } else {
                                    von    += ' '+vonlastarray[j];
                                }
                            } else {
                                von    += ' '+vonlastarray[j];
                            }
                        }
                    }
                    last += ' '+vonlastarray[size-1];
                }
                //Now we check if it is version three (three entries in the array (two commas)
                if (3==tmparray.length) {
                    jr = tmparray[1];
                }
                //Everything in the last entry is first
                first = tmparray[tmparray.length-1];
            }
            authorarray[i] = {'first':trim(first), 'von':trim(von), 'last':trim(last), 'jr':trim(jr)};
        }
        return authorarray;
    },

    /**
     * Case Determination according to the needs of BibTex
     *
     * To parse the Author(s) correctly a determination is needed
     * to get the Case of a word+ There are three possible values:
     * - Upper Case (return value 1)
     * - Lower Case (return value 0)
     * - Caseless   (return value -1)
     *
     * @access private
     * @param string word
     * @return int The Case or PEAR_Error if there was a problem
     */
    '_determineCase': function(word) {
        var ret         = -1;
        var trimmedword = trim (word);
        /*We need this variable+ Without the next of would not work
         (trim changes the variable automatically to a string!)*/
        if (is_string(word) && (strlen(trimmedword) > 0)) {
            var i         = 0;
            var found     = false;
            var openbrace = 0;
            while (!found && (i <= strlen(word))) {
                var letter = substr(trimmedword, i, 1);
                var ordv    = ord(letter);
                if (ordv == 123) { //Open brace
                    openbrace++;
                }
                if (ordv == 125) { //Closing brace
                    openbrace--;
                }
                if ((ordv>=65) && (ordv<=90) && (0==openbrace)) { //The first character is uppercase
                    ret   = 1;
                    found = true;
                } else if ( (ordv>=97) && (ordv<=122) && (0==openbrace) ) { //The first character is lowercase
                    ret   = 0;
                    found = true;
                } else { //Not yet found
                    i++;
                }
            }
        } else {
            ret = this.raiseError('Could not determine case on word: '+word);
        }
        return ret;
    },


    'isError': function(obj){
    	return ( typeof(obj) == 'Object' && obj.isError == 1 );

    },

    /**
     * Validation of a value
     *
     * There may be several problems with the value of a field+
     * These problems exist but do not break the parsing+
     * If a problem is detected a warning is appended to the array warnings+
     *
     * @access private
     * @param string entry The entry aka one line which which should be validated
     * @param string wholeentry The whole BibTex Entry which the one line is part of
     * @return void
     */
    '_validateValue': function(entry, wholeentry)
    {
        //There is no @ allowed if the entry is enclosed by braces
        if ( entry.match(/^{.*@.*}/)) {
            this._generateWarning('WARNING_AT_IN_BRACES', entry, wholeentry);
        }
        //No escaped " allowed if the entry is enclosed by double quotes
        if (entry.match(/^\".*\\".*\"/)) {
            this._generateWarning('WARNING_ESCAPED_DOUBLE_QUOTE_INSIDE_DOUBLE_QUOTES', entry, wholeentry);
        }
        //Amount of Braces is not correct
        var open     = 0;
        var lastchar = '';
        var charv     = '';
        for (var i = 0; i < strlen(entry); i++) {
            charv = substr(entry, i, 1);
            if (('{' == charv) && ('\\' != lastchar)) {
                open++;
            }
            if (('}' == charv) && ('\\' != lastchar)) {
                open--;
            }
            lastchar = charv;
        }
        if (0 != open) {
            this._generateWarning('WARNING_UNBALANCED_AMOUNT_OF_BRACES', entry, wholeentry);
        }
    },

    /**
     * Remove curly braces from entry
     *
     * @access private
     * @param string value The value in which curly braces to be removed
     * @param string Value with removed curly braces
     */
    '_removeCurlyBraces': function(value)
    {
        //First we save the delimiters
        var beginningdels = array_keys(this._delimiters);
        var firstchar     = substr(value, 0, 1);
        var lastchar      = substr(value, -1, 1);
        var begin         = '';
        var end           = '';
        while (in_array(firstchar, beginningdels)) { //The first character is an opening delimiter
            if (lastchar == this._delimiters[firstchar]) { //Matches to closing Delimiter
                begin += firstchar;
                end   += lastchar;
                value  = substr(value, 1, -1);
            } else {
                break;
            }
            firstchar = substr(value, 0, 1);
            lastchar  = substr(value, -1, 1);
        }
        //Now we get rid of the curly braces
        var pattern     = '/([^\\\\])\{(+*?[^\\\\])\}/';
        var replacement = '12';
        value       = value.replace(/([^\\\\])\{(.*?[^\\\\])\}/, replacement);
        //Reattach delimiters
        value       = begin+value+end;
        return value;
    },

    /**
     * Generates a warning
     *
     * @access private
     * @param string type The type of the warning
     * @param string entry The line of the entry where the warning occurred
     * @param string wholeentry OPTIONAL The whole entry where the warning occurred
     */
    '_generateWarning': function(type, entry, wholeentry)
    {
    	if ( typeof wholeentry == 'undefined' ) wholeentry = '';
        var warning = {};
        warning['warning']    = type;
        warning['entry']      = entry;
        warning['wholeentry'] = wholeentry;
        this.warnings[this.warnings.length]      = warning;
    },

    /**
     * Cleares all warnings
     *
     * @access public
     */
    'clearWarnings': function()
    {
        this.warnings = array();
    },

    /**
     * Is there a warning?
     *
     * @access public
     * @return true if there is, false otherwise
     */
    'hasWarning': function()
    {
        if (sizeof(this.warnings)>0) return true;
        else return false;
    },

    /**
     * Returns the amount of available BibTex entries
     *
     * @access public
     * @return int The amount of available BibTex entries
     */
    'amount': function()
    {
        return sizeof(this.data);
    },
    /**
     * Returns the author formatted
     *
     * The Author is formatted as setted in the authorstring
     *
     * @access private
     * @param array array Author array
     * @return string the formatted author string
     */
    '_formatAuthor': function(array)
    {
        if (!array_key_exists('von', array)) {
            array['von'] = '';
        } else {
            array['von'] = trim(array['von']);
        }
        if (!array_key_exists('last', array)) {
            array['last'] = '';
        } else {
            array['last'] = trim(array['last']);
        }
        if (!array_key_exists('jr', array)) {
            array['jr'] = '';
        } else {
            array['jr'] = trim(array['jr']);
        }
        if (!array_key_exists('first', array)) {
            array['first'] = '';
        } else {
            array['first'] = trim(array['first']);
        }
        ret = this.authorstring;
        ret = str_replace("VON", array['von'], ret);
        ret = str_replace("LAST", array['last'], ret);
        ret = str_replace("JR", array['jr'], ret);
        ret = str_replace("FIRST", array['first'], ret);
        return trim(ret);
    },

    /**
     * Converts the stored BibTex entries to a BibTex String
     *
     * In the field list, the author is the last field+
     *
     * @access public
     * @return string The BibTex string
     */
    'bibTex': function()
    {
        var bibtex = '';
        for (var i=0 ; i<this.data.length; i++) {
        	var entry = this.data[i];
            //Intro
            bibtex += '@'+strtolower(entry['entryType'])+' { '+entry['cite']+",\n";
            //Other fields except author
            for (key in entry) {
            	var val = entry[key];
                if (this._options['wordWrapWidth']>0) {
                    val = this._wordWrap(val);
                }
                if (!in_array(key, array('cite','entryType','author'))) {
                    bibtex += "\t"+key+' = {'+val+"},\n";
                }
            }
            //Author
            if (array_key_exists('author', entry)) {
                if (this._options['extractAuthors']) {
                    tmparray = []; //In this array the authors are saved and the joind with an and
                    for (j in entry['author']) {
                    	var authorentry = entry['author'][j];
                        tmparray[tmparray.length] = this._formatAuthor(authorentry);
                    }
                    author = join(' and ', tmparray);
                } else {
                    author = entry['author'];
                }
            } else {
                author = '';
            }
            bibtex += "\tauthor = {"+author+"}\n";
            bibtex+="}\n\n";
        }
        return bibtex;
    },

    /**
     * Adds a new BibTex entry to the data
     *
     * @access public
     * @param array newentry The new data to add
     * @return void
     */
    'addEntry': function(newentry)
    {
        this.data[this.data.length] = newentry;
    },

    /**
     * Returns statistic
     *
     * This functions returns a hash table+ The keys are the different
     * entry types and the values are the amount of these entries+
     *
     * @access public
     * @return array Hash Table with the data
     */
    'getStatistic': function()
    {
        var ret = array();
        for (var i=0; i<this.data.length; i++) {
        	var entry = this.data[i];
            if (array_key_exists(entry['entryType'], ret)) {
                ret[entry['entryType']]++;
            } else {
                ret[entry['entryType']] = 1;
            }
        }
        return ret;
    },

    /**
     * Returns the stored data in RTF format
     *
     * This method simply returns a RTF formatted string+ This is done very
     * simple and is not intended for heavy using and fine formatting+ This
     * should be done by BibTex! It is intended to give some kind of quick
     * preview or to send someone a reference list as word/rtf format (even
     * some people in the scientific field still use word)+ If you want to
     * change the default format you have to override the class variable
     * "rtfstring"+ This variable is used and the placeholders simply replaced+
     * Lines with no data cause an warning!
     *
     * @return string the RTF Strings
     */
    'rtf': function()
    {
        var ret = "{\\rtf\n";
        for (var i=0; i<this.data.length; i++) {
        	var entry = this.data[i];
            line    = this.rtfstring;
            title   = '';
            journal = '';
            year    = '';
            authors = '';
            if (array_key_exists('title', entry)) {
                title = this._unwrap(entry['title']);
            }
            if (array_key_exists('journal', entry)) {
                journal = this._unwrap(entry['journal']);
            }
            if (array_key_exists('year', entry)) {
                year = this._unwrap(entry['year']);
            }
            if (array_key_exists('author', entry)) {
                if (this._options['extractAuthors']) {
                    tmparray = []; //In this array the authors are saved and the joind with an and
                    for (var j in entry['author']) {
                    	var authorentry = entry['author'][j];
                        tmparray[tmparray.length] = this._formatAuthor(authorentry);
                    }
                    authors = join(', ', tmparray);
                } else {
                    authors = entry['author'];
                }
            }
            if ((''!=title) || (''!=journal) || (''!=year) || (''!=authors)) {
                line = str_replace("TITLE", title, line);
                line = str_replace("JOURNAL", journal, line);
                line = str_replace("YEAR", year, line);
                line = str_replace("AUTHORS", authors, line);
                line += "\n\\par\n";
                ret  += line;
            } else {
                this._generateWarning('WARNING_LINE_WAS_NOT_CONVERTED', '', print_r(entry,1));
            }
        }
        ret += '}';
        return ret;
    },

    /**
     * Returns the stored data in HTML format
     *
     * This method simply returns a HTML formatted string+ This is done very
     * simple and is not intended for heavy using and fine formatting+ This
     * should be done by BibTex! It is intended to give some kind of quick
     * preview+ If you want to change the default format you have to override
     * the class variable "htmlstring"+ This variable is used and the placeholders
     * simply replaced+
     * Lines with no data cause an warning!
     *
     * @return string the HTML Strings
     */
    'html': function(min,max)
    {
    	if ( typeof min == 'undefined' ) min = 0;
    	if ( typeof max == 'undefined' ) max = this.data.length;
        var ret = "<p>\n";
        for (var i =min; i<max; i++){
        	var entry = this.data[i];
            var line    = this.htmlstring;
            var title   = '';
            var journal = '';
            var year    = '';
            var authors = '';
            if (array_key_exists('title', entry)) {
                title = this._unwrap(entry['title']);
            }
            if (array_key_exists('journal', entry)) {
                journal = this._unwrap(entry['journal']);
            }
            if (array_key_exists('year', entry)) {
                year = this._unwrap(entry['year']);
            }
            if (array_key_exists('author', entry)) {
                if (this._options['extractAuthors']) {
                    tmparray = []; //In this array the authors are saved and the joind with an and
                    for (j in entry['author'] ) {
                    	var authorentry = entry['author'][j];
                        tmparray[tmparray.length] = this._formatAuthor(authorentry);
                    }
                    authors = join(', ', tmparray);
                } else {
                    authors = entry['author'];
                }
            }

            if ((''!=title) || (''!=journal) || (''!=year) || (''!=authors)) {
                line = str_replace("TITLE", title, line);
                line = str_replace("JOURNAL", journal, line);
                line = str_replace("YEAR", year, line);
                line = str_replace("AUTHORS", authors, line);
                line += "\n";
                ret  += line;
            } else {
                this._generateWarning('WARNING_LINE_WAS_NOT_CONVERTED', '', print_r(entry,1));
            }
        }
        ret += "</p>\n";
        return ret;
    }
};/*!
 * File:        jquery.dataTables.min.js
 * Version:     1.6.2
 * Author:      Allan Jardine (www.sprymedia.co.uk)
 * Info:        www.datatables.net
 *
 * Copyright 2008-2010 Allan Jardine, all rights reserved.
 *
 * This source file is free software, under either the GPL v2 license or a
 * BSD style license, as supplied with this software.
 *
 * This source file is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the license files for details.
 */
(function($){$.fn.dataTableSettings=[];var _aoSettings=$.fn.dataTableSettings;$.fn.dataTableExt={};
var _oExt=$.fn.dataTableExt;_oExt.sVersion="1.6.2";_oExt.iApiIndex=0;_oExt.oApi={};
_oExt.afnFiltering=[];_oExt.aoFeatures=[];_oExt.ofnSearch={};_oExt.afnSortData=[];
_oExt.oStdClasses={sPagePrevEnabled:"paginate_enabled_previous",sPagePrevDisabled:"paginate_disabled_previous",sPageNextEnabled:"paginate_enabled_next",sPageNextDisabled:"paginate_disabled_next",sPageJUINext:"",sPageJUIPrev:"",sPageButton:"paginate_button",sPageButtonActive:"paginate_active",sPageButtonStaticDisabled:"paginate_button",sPageFirst:"first",sPagePrevious:"previous",sPageNext:"next",sPageLast:"last",sStripOdd:"odd",sStripEven:"even",sRowEmpty:"dataTables_empty",sWrapper:"dataTables_wrapper",sFilter:"dataTables_filter",sInfo:"dataTables_info",sPaging:"dataTables_paginate paging_",sLength:"dataTables_length",sProcessing:"dataTables_processing",sSortAsc:"sorting_asc",sSortDesc:"sorting_desc",sSortable:"sorting",sSortableAsc:"sorting_asc_disabled",sSortableDesc:"sorting_desc_disabled",sSortableNone:"sorting_disabled",sSortColumn:"sorting_",sSortJUIAsc:"",sSortJUIDesc:"",sSortJUI:"",sSortJUIAscAllowed:"",sSortJUIDescAllowed:""};
_oExt.oJUIClasses={sPagePrevEnabled:"fg-button ui-state-default ui-corner-left",sPagePrevDisabled:"fg-button ui-state-default ui-corner-left ui-state-disabled",sPageNextEnabled:"fg-button ui-state-default ui-corner-right",sPageNextDisabled:"fg-button ui-state-default ui-corner-right ui-state-disabled",sPageJUINext:"ui-icon ui-icon-circle-arrow-e",sPageJUIPrev:"ui-icon ui-icon-circle-arrow-w",sPageButton:"fg-button ui-state-default",sPageButtonActive:"fg-button ui-state-default ui-state-disabled",sPageButtonStaticDisabled:"fg-button ui-state-default ui-state-disabled",sPageFirst:"first ui-corner-tl ui-corner-bl",sPagePrevious:"previous",sPageNext:"next",sPageLast:"last ui-corner-tr ui-corner-br",sStripOdd:"odd",sStripEven:"even",sRowEmpty:"dataTables_empty",sWrapper:"dataTables_wrapper",sFilter:"dataTables_filter",sInfo:"dataTables_info",sPaging:"dataTables_paginate fg-buttonset fg-buttonset-multi paging_",sLength:"dataTables_length",sProcessing:"dataTables_processing",sSortAsc:"ui-state-default",sSortDesc:"ui-state-default",sSortable:"ui-state-default",sSortableAsc:"ui-state-default",sSortableDesc:"ui-state-default",sSortableNone:"ui-state-default",sSortColumn:"sorting_",sSortJUIAsc:"css_right ui-icon ui-icon-triangle-1-n",sSortJUIDesc:"css_right ui-icon ui-icon-triangle-1-s",sSortJUI:"css_right ui-icon ui-icon-carat-2-n-s",sSortJUIAscAllowed:"css_right ui-icon ui-icon-carat-1-n",sSortJUIDescAllowed:"css_right ui-icon ui-icon-carat-1-s"};
_oExt.oPagination={two_button:{fnInit:function(oSettings,nPaging,fnCallbackDraw){var nPrevious,nNext,nPreviousInner,nNextInner;
if(!oSettings.bJUI){nPrevious=document.createElement("div");nNext=document.createElement("div")
}else{nPrevious=document.createElement("a");nNext=document.createElement("a");nNextInner=document.createElement("span");
nNextInner.className=oSettings.oClasses.sPageJUINext;nNext.appendChild(nNextInner);
nPreviousInner=document.createElement("span");nPreviousInner.className=oSettings.oClasses.sPageJUIPrev;
nPrevious.appendChild(nPreviousInner)}nPrevious.className=oSettings.oClasses.sPagePrevDisabled;
nNext.className=oSettings.oClasses.sPageNextDisabled;nPrevious.title=oSettings.oLanguage.oPaginate.sPrevious;
nNext.title=oSettings.oLanguage.oPaginate.sNext;nPaging.appendChild(nPrevious);nPaging.appendChild(nNext);
$(nPrevious).click(function(){if(oSettings.oApi._fnPageChange(oSettings,"previous")){fnCallbackDraw(oSettings)
}});$(nNext).click(function(){if(oSettings.oApi._fnPageChange(oSettings,"next")){fnCallbackDraw(oSettings)
}});$(nPrevious).bind("selectstart",function(){return false});$(nNext).bind("selectstart",function(){return false
});if(oSettings.sTableId!==""&&typeof oSettings.aanFeatures.p=="undefined"){nPaging.setAttribute("id",oSettings.sTableId+"_paginate");
nPrevious.setAttribute("id",oSettings.sTableId+"_previous");nNext.setAttribute("id",oSettings.sTableId+"_next")
}},fnUpdate:function(oSettings,fnCallbackDraw){if(!oSettings.aanFeatures.p){return
}var an=oSettings.aanFeatures.p;for(var i=0,iLen=an.length;i<iLen;i++){if(an[i].childNodes.length!==0){an[i].childNodes[0].className=(oSettings._iDisplayStart===0)?oSettings.oClasses.sPagePrevDisabled:oSettings.oClasses.sPagePrevEnabled;
an[i].childNodes[1].className=(oSettings.fnDisplayEnd()==oSettings.fnRecordsDisplay())?oSettings.oClasses.sPageNextDisabled:oSettings.oClasses.sPageNextEnabled
}}}},iFullNumbersShowPages:5,full_numbers:{fnInit:function(oSettings,nPaging,fnCallbackDraw){var nFirst=document.createElement("span");
var nPrevious=document.createElement("span");var nList=document.createElement("span");
var nNext=document.createElement("span");var nLast=document.createElement("span");
nFirst.innerHTML=oSettings.oLanguage.oPaginate.sFirst;nPrevious.innerHTML=oSettings.oLanguage.oPaginate.sPrevious;
nNext.innerHTML=oSettings.oLanguage.oPaginate.sNext;nLast.innerHTML=oSettings.oLanguage.oPaginate.sLast;
var oClasses=oSettings.oClasses;nFirst.className=oClasses.sPageButton+" "+oClasses.sPageFirst;
nPrevious.className=oClasses.sPageButton+" "+oClasses.sPagePrevious;nNext.className=oClasses.sPageButton+" "+oClasses.sPageNext;
nLast.className=oClasses.sPageButton+" "+oClasses.sPageLast;nPaging.appendChild(nFirst);
nPaging.appendChild(nPrevious);nPaging.appendChild(nList);nPaging.appendChild(nNext);
nPaging.appendChild(nLast);$(nFirst).click(function(){if(oSettings.oApi._fnPageChange(oSettings,"first")){fnCallbackDraw(oSettings)
}});$(nPrevious).click(function(){if(oSettings.oApi._fnPageChange(oSettings,"previous")){fnCallbackDraw(oSettings)
}});$(nNext).click(function(){if(oSettings.oApi._fnPageChange(oSettings,"next")){fnCallbackDraw(oSettings)
}});$(nLast).click(function(){if(oSettings.oApi._fnPageChange(oSettings,"last")){fnCallbackDraw(oSettings)
}});$("span",nPaging).bind("mousedown",function(){return false}).bind("selectstart",function(){return false
});if(oSettings.sTableId!==""&&typeof oSettings.aanFeatures.p=="undefined"){nPaging.setAttribute("id",oSettings.sTableId+"_paginate");
nFirst.setAttribute("id",oSettings.sTableId+"_first");nPrevious.setAttribute("id",oSettings.sTableId+"_previous");
nNext.setAttribute("id",oSettings.sTableId+"_next");nLast.setAttribute("id",oSettings.sTableId+"_last")
}},fnUpdate:function(oSettings,fnCallbackDraw){if(!oSettings.aanFeatures.p){return
}var iPageCount=_oExt.oPagination.iFullNumbersShowPages;var iPageCountHalf=Math.floor(iPageCount/2);
var iPages=Math.ceil((oSettings.fnRecordsDisplay())/oSettings._iDisplayLength);var iCurrentPage=Math.ceil(oSettings._iDisplayStart/oSettings._iDisplayLength)+1;
var sList="";var iStartButton,iEndButton,i,iLen;var oClasses=oSettings.oClasses;if(iPages<iPageCount){iStartButton=1;
iEndButton=iPages}else{if(iCurrentPage<=iPageCountHalf){iStartButton=1;iEndButton=iPageCount
}else{if(iCurrentPage>=(iPages-iPageCountHalf)){iStartButton=iPages-iPageCount+1;
iEndButton=iPages}else{iStartButton=iCurrentPage-Math.ceil(iPageCount/2)+1;iEndButton=iStartButton+iPageCount-1
}}}for(i=iStartButton;i<=iEndButton;i++){if(iCurrentPage!=i){sList+='<span class="'+oClasses.sPageButton+'">'+i+"</span>"
}else{sList+='<span class="'+oClasses.sPageButtonActive+'">'+i+"</span>"}}var an=oSettings.aanFeatures.p;
var anButtons,anStatic,nPaginateList;var fnClick=function(){var iTarget=(this.innerHTML*1)-1;
oSettings._iDisplayStart=iTarget*oSettings._iDisplayLength;fnCallbackDraw(oSettings);
return false};var fnFalse=function(){return false};for(i=0,iLen=an.length;i<iLen;
i++){if(an[i].childNodes.length===0){continue}nPaginateList=an[i].childNodes[2];nPaginateList.innerHTML=sList;
$("span",nPaginateList).click(fnClick).bind("mousedown",fnFalse).bind("selectstart",fnFalse);
anButtons=an[i].getElementsByTagName("span");anStatic=[anButtons[0],anButtons[1],anButtons[anButtons.length-2],anButtons[anButtons.length-1]];
$(anStatic).removeClass(oClasses.sPageButton+" "+oClasses.sPageButtonActive+" "+oClasses.sPageButtonStaticDisabled);
if(iCurrentPage==1){anStatic[0].className+=" "+oClasses.sPageButtonStaticDisabled;
anStatic[1].className+=" "+oClasses.sPageButtonStaticDisabled}else{anStatic[0].className+=" "+oClasses.sPageButton;
anStatic[1].className+=" "+oClasses.sPageButton}if(iPages===0||iCurrentPage==iPages||oSettings._iDisplayLength==-1){anStatic[2].className+=" "+oClasses.sPageButtonStaticDisabled;
anStatic[3].className+=" "+oClasses.sPageButtonStaticDisabled}else{anStatic[2].className+=" "+oClasses.sPageButton;
anStatic[3].className+=" "+oClasses.sPageButton}}}}};_oExt.oSort={"string-asc":function(a,b){var x=a.toLowerCase();
var y=b.toLowerCase();return((x<y)?-1:((x>y)?1:0))},"string-desc":function(a,b){var x=a.toLowerCase();
var y=b.toLowerCase();return((x<y)?1:((x>y)?-1:0))},"html-asc":function(a,b){var x=a.replace(/<.*?>/g,"").toLowerCase();
var y=b.replace(/<.*?>/g,"").toLowerCase();return((x<y)?-1:((x>y)?1:0))},"html-desc":function(a,b){var x=a.replace(/<.*?>/g,"").toLowerCase();
var y=b.replace(/<.*?>/g,"").toLowerCase();return((x<y)?1:((x>y)?-1:0))},"date-asc":function(a,b){var x=Date.parse(a);
var y=Date.parse(b);if(isNaN(x)){x=Date.parse("01/01/1970 00:00:00")}if(isNaN(y)){y=Date.parse("01/01/1970 00:00:00")
}return x-y},"date-desc":function(a,b){var x=Date.parse(a);var y=Date.parse(b);if(isNaN(x)){x=Date.parse("01/01/1970 00:00:00")
}if(isNaN(y)){y=Date.parse("01/01/1970 00:00:00")}return y-x},"numeric-asc":function(a,b){var x=a=="-"?0:a;
var y=b=="-"?0:b;return x-y},"numeric-desc":function(a,b){var x=a=="-"?0:a;var y=b=="-"?0:b;
return y-x}};_oExt.aTypes=[function(sData){if(typeof sData=="number"){return"numeric"
}else{if(typeof sData.charAt!="function"){return null}}var sValidFirstChars="0123456789-";
var sValidChars="0123456789.";var Char;var bDecimal=false;Char=sData.charAt(0);if(sValidFirstChars.indexOf(Char)==-1){return null
}for(var i=1;i<sData.length;i++){Char=sData.charAt(i);if(sValidChars.indexOf(Char)==-1){return null
}if(Char=="."){if(bDecimal){return null}bDecimal=true}}return"numeric"},function(sData){var iParse=Date.parse(sData);
if(iParse!==null&&!isNaN(iParse)){return"date"}return null}];_oExt._oExternConfig={iNextUnique:0};
$.fn.dataTable=function(oInit){function classSettings(){this.fnRecordsTotal=function(){if(this.oFeatures.bServerSide){return this._iRecordsTotal
}else{return this.aiDisplayMaster.length}};this.fnRecordsDisplay=function(){if(this.oFeatures.bServerSide){return this._iRecordsDisplay
}else{return this.aiDisplay.length}};this.fnDisplayEnd=function(){if(this.oFeatures.bServerSide){return this._iDisplayStart+this.aiDisplay.length
}else{return this._iDisplayEnd}};this.sInstance=null;this.oFeatures={bPaginate:true,bLengthChange:true,bFilter:true,bSort:true,bInfo:true,bAutoWidth:true,bProcessing:false,bSortClasses:true,bStateSave:false,bServerSide:false};
this.aanFeatures=[];this.oLanguage={sProcessing:"Processing...",sLengthMenu:"Show _MENU_ entries",sZeroRecords:"No matching records found",sInfo:"Showing _START_ to _END_ of _TOTAL_ entries",sInfoEmpty:"Showing 0 to 0 of 0 entries",sInfoFiltered:"(filtered from _MAX_ total entries)",sInfoPostFix:"",sSearch:"Search:",sUrl:"",oPaginate:{sFirst:"First",sPrevious:"Previous",sNext:"Next",sLast:"Last"}};
this.aoData=[];this.aiDisplay=[];this.aiDisplayMaster=[];this.aoColumns=[];this.iNextId=0;
this.asDataSearch=[];this.oPreviousSearch={sSearch:"",bEscapeRegex:true};this.aoPreSearchCols=[];
this.aaSorting=[[0,"asc",0]];this.aaSortingFixed=null;this.asStripClasses=[];this.fnRowCallback=null;
this.fnHeaderCallback=null;this.fnFooterCallback=null;this.aoDrawCallback=[];this.fnInitComplete=null;
this.sTableId="";this.nTable=null;this.iDefaultSortIndex=0;this.bInitialised=false;
this.aoOpenRows=[];this.sDom="lfrtip";this.sPaginationType="two_button";this.iCookieDuration=60*60*2;
this.sAjaxSource=null;this.bAjaxDataGet=true;this.fnServerData=$.getJSON;this.iServerDraw=0;
this._iDisplayLength=10;this._iDisplayStart=0;this._iDisplayEnd=10;this._iRecordsTotal=0;
this._iRecordsDisplay=0;this.bJUI=false;this.oClasses=_oExt.oStdClasses;this.bFiltered=false;
this.bSorted=false}this.oApi={};this.fnDraw=function(bComplete){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
if(typeof bComplete!="undefined"&&bComplete===false){_fnCalculateEnd(oSettings);_fnDraw(oSettings)
}else{_fnReDraw(oSettings)}};this.fnFilter=function(sInput,iColumn,bEscapeRegex){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
if(typeof bEscapeRegex=="undefined"){bEscapeRegex=true}if(typeof iColumn=="undefined"||iColumn===null){_fnFilterComplete(oSettings,{sSearch:sInput,bEscapeRegex:bEscapeRegex},1)
}else{oSettings.aoPreSearchCols[iColumn].sSearch=sInput;oSettings.aoPreSearchCols[iColumn].bEscapeRegex=bEscapeRegex;
_fnFilterComplete(oSettings,oSettings.oPreviousSearch,1)}};this.fnSettings=function(nNode){return _fnSettingsFromNode(this[_oExt.iApiIndex])
};this.fnVersionCheck=function(sVersion){var fnZPad=function(Zpad,count){while(Zpad.length<count){Zpad+="0"
}return Zpad};var aThis=_oExt.sVersion.split(".");var aThat=sVersion.split(".");var sThis="",sThat="";
for(var i=0,iLen=aThat.length;i<iLen;i++){sThis+=fnZPad(aThis[i],3);sThat+=fnZPad(aThat[i],3)
}return parseInt(sThis,10)>=parseInt(sThat,10)};this.fnSort=function(aaSort){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
oSettings.aaSorting=aaSort;_fnSort(oSettings)};this.fnSortListener=function(nNode,iColumn,fnCallback){_fnSortAttachListener(_fnSettingsFromNode(this[_oExt.iApiIndex]),nNode,iColumn,fnCallback)
};this.fnAddData=function(mData,bRedraw){if(mData.length===0){return[]}var aiReturn=[];
var iTest;var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);if(typeof mData[0]=="object"){for(var i=0;
i<mData.length;i++){iTest=_fnAddData(oSettings,mData[i]);if(iTest==-1){return aiReturn
}aiReturn.push(iTest)}}else{iTest=_fnAddData(oSettings,mData);if(iTest==-1){return aiReturn
}aiReturn.push(iTest)}oSettings.aiDisplay=oSettings.aiDisplayMaster.slice();_fnBuildSearchArray(oSettings,1);
if(typeof bRedraw=="undefined"||bRedraw){_fnReDraw(oSettings)}return aiReturn};this.fnDeleteRow=function(mTarget,fnCallBack,bNullRow){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
var i,iAODataIndex;iAODataIndex=(typeof mTarget=="object")?_fnNodeToDataIndex(oSettings,mTarget):mTarget;
for(i=0;i<oSettings.aiDisplayMaster.length;i++){if(oSettings.aiDisplayMaster[i]==iAODataIndex){oSettings.aiDisplayMaster.splice(i,1);
break}}for(i=0;i<oSettings.aiDisplay.length;i++){if(oSettings.aiDisplay[i]==iAODataIndex){oSettings.aiDisplay.splice(i,1);
break}}_fnBuildSearchArray(oSettings,1);if(typeof fnCallBack=="function"){fnCallBack.call(this)
}if(oSettings._iDisplayStart>=oSettings.aiDisplay.length){oSettings._iDisplayStart-=oSettings._iDisplayLength;
if(oSettings._iDisplayStart<0){oSettings._iDisplayStart=0}}_fnCalculateEnd(oSettings);
_fnDraw(oSettings);var aData=oSettings.aoData[iAODataIndex]._aData.slice();if(typeof bNullRow!="undefined"&&bNullRow===true){oSettings.aoData[iAODataIndex]=null
}return aData};this.fnClearTable=function(bRedraw){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
_fnClearTable(oSettings);if(typeof bRedraw=="undefined"||bRedraw){_fnDraw(oSettings)
}};this.fnOpen=function(nTr,sHtml,sClass){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
this.fnClose(nTr);var nNewRow=document.createElement("tr");var nNewCell=document.createElement("td");
nNewRow.appendChild(nNewCell);nNewCell.className=sClass;nNewCell.colSpan=_fnVisbleColumns(oSettings);
nNewCell.innerHTML=sHtml;var nTrs=$("tbody tr",oSettings.nTable);if($.inArray(nTr,nTrs)!=-1){$(nNewRow).insertAfter(nTr)
}if(!oSettings.oFeatures.bServerSide){oSettings.aoOpenRows.push({nTr:nNewRow,nParent:nTr})
}return nNewRow};this.fnClose=function(nTr){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
for(var i=0;i<oSettings.aoOpenRows.length;i++){if(oSettings.aoOpenRows[i].nParent==nTr){var nTrParent=oSettings.aoOpenRows[i].nTr.parentNode;
if(nTrParent){nTrParent.removeChild(oSettings.aoOpenRows[i].nTr)}oSettings.aoOpenRows.splice(i,1);
return 0}}return 1};this.fnGetData=function(mRow){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
if(typeof mRow!="undefined"){var iRow=(typeof mRow=="object")?_fnNodeToDataIndex(oSettings,mRow):mRow;
return oSettings.aoData[iRow]._aData}return _fnGetDataMaster(oSettings)};this.fnGetNodes=function(iRow){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
if(typeof iRow!="undefined"){return oSettings.aoData[iRow].nTr}return _fnGetTrNodes(oSettings)
};this.fnGetPosition=function(nNode){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
var i;if(nNode.nodeName=="TR"){return _fnNodeToDataIndex(oSettings,nNode)}else{if(nNode.nodeName=="TD"){var iDataIndex=_fnNodeToDataIndex(oSettings,nNode.parentNode);
var iCorrector=0;for(var j=0;j<oSettings.aoColumns.length;j++){if(oSettings.aoColumns[j].bVisible){if(oSettings.aoData[iDataIndex].nTr.getElementsByTagName("td")[j-iCorrector]==nNode){return[iDataIndex,j-iCorrector,j]
}}else{iCorrector++}}}}return null};this.fnUpdate=function(mData,mRow,iColumn,bRedraw){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
var iVisibleColumn;var sDisplay;var iRow=(typeof mRow=="object")?_fnNodeToDataIndex(oSettings,mRow):mRow;
if(typeof mData!="object"){sDisplay=mData;oSettings.aoData[iRow]._aData[iColumn]=sDisplay;
if(oSettings.aoColumns[iColumn].fnRender!==null){sDisplay=oSettings.aoColumns[iColumn].fnRender({iDataRow:iRow,iDataColumn:iColumn,aData:oSettings.aoData[iRow]._aData,oSettings:oSettings});
if(oSettings.aoColumns[iColumn].bUseRendered){oSettings.aoData[iRow]._aData[iColumn]=sDisplay
}}iVisibleColumn=_fnColumnIndexToVisible(oSettings,iColumn);if(iVisibleColumn!==null){oSettings.aoData[iRow].nTr.getElementsByTagName("td")[iVisibleColumn].innerHTML=sDisplay
}}else{if(mData.length!=oSettings.aoColumns.length){alert("DataTables warning: An array passed to fnUpdate must have the same number of columns as the table in question - in this case "+oSettings.aoColumns.length);
return 1}for(var i=0;i<mData.length;i++){sDisplay=mData[i];oSettings.aoData[iRow]._aData[i]=sDisplay;
if(oSettings.aoColumns[i].fnRender!==null){sDisplay=oSettings.aoColumns[i].fnRender({iDataRow:iRow,iDataColumn:i,aData:oSettings.aoData[iRow]._aData,oSettings:oSettings});
if(oSettings.aoColumns[i].bUseRendered){oSettings.aoData[iRow]._aData[i]=sDisplay
}}iVisibleColumn=_fnColumnIndexToVisible(oSettings,i);if(iVisibleColumn!==null){oSettings.aoData[iRow].nTr.getElementsByTagName("td")[iVisibleColumn].innerHTML=sDisplay
}}}_fnBuildSearchArray(oSettings,1);if(typeof bRedraw!="undefined"&&bRedraw){_fnReDraw(oSettings)
}return 0};this.fnSetColumnVis=function(iCol,bShow){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
var i,iLen;var iColumns=oSettings.aoColumns.length;var nTd,anTds;if(oSettings.aoColumns[iCol].bVisible==bShow){return
}var nTrHead=$("thead:eq(0)>tr",oSettings.nTable)[0];var nTrFoot=$("tfoot:eq(0)>tr",oSettings.nTable)[0];
var anTheadTh=[];var anTfootTh=[];for(i=0;i<iColumns;i++){anTheadTh.push(oSettings.aoColumns[i].nTh);
anTfootTh.push(oSettings.aoColumns[i].nTf)}if(bShow){var iInsert=0;for(i=0;i<iCol;
i++){if(oSettings.aoColumns[i].bVisible){iInsert++}}if(iInsert>=_fnVisbleColumns(oSettings)){nTrHead.appendChild(anTheadTh[iCol]);
if(nTrFoot){nTrFoot.appendChild(anTfootTh[iCol])}for(i=0,iLen=oSettings.aoData.length;
i<iLen;i++){nTd=oSettings.aoData[i]._anHidden[iCol];oSettings.aoData[i].nTr.appendChild(nTd)
}}else{var iBefore;for(i=iCol;i<iColumns;i++){iBefore=_fnColumnIndexToVisible(oSettings,i);
if(iBefore!==null){break}}nTrHead.insertBefore(anTheadTh[iCol],nTrHead.getElementsByTagName("th")[iBefore]);
if(nTrFoot){nTrFoot.insertBefore(anTfootTh[iCol],nTrFoot.getElementsByTagName("th")[iBefore])
}anTds=_fnGetTdNodes(oSettings);for(i=0,iLen=oSettings.aoData.length;i<iLen;i++){nTd=oSettings.aoData[i]._anHidden[iCol];
oSettings.aoData[i].nTr.insertBefore(nTd,$(">td:eq("+iBefore+")",oSettings.aoData[i].nTr)[0])
}}oSettings.aoColumns[iCol].bVisible=true}else{nTrHead.removeChild(anTheadTh[iCol]);
if(nTrFoot){nTrFoot.removeChild(anTfootTh[iCol])}anTds=_fnGetTdNodes(oSettings);for(i=0,iLen=oSettings.aoData.length;
i<iLen;i++){nTd=anTds[(i*oSettings.aoColumns.length)+iCol];oSettings.aoData[i]._anHidden[iCol]=nTd;
nTd.parentNode.removeChild(nTd)}oSettings.aoColumns[iCol].bVisible=false}for(i=0,iLen=oSettings.aoOpenRows.length;
i<iLen;i++){oSettings.aoOpenRows[i].nTr.colSpan=_fnVisbleColumns(oSettings)}_fnSaveState(oSettings)
};this.fnPageChange=function(sAction,bRedraw){var oSettings=_fnSettingsFromNode(this[_oExt.iApiIndex]);
_fnPageChange(oSettings,sAction);_fnCalculateEnd(oSettings);if(typeof bRedraw=="undefined"||bRedraw){_fnDraw(oSettings)
}};function _fnExternApiFunc(sFunc){return function(){var aArgs=[_fnSettingsFromNode(this[_oExt.iApiIndex])].concat(Array.prototype.slice.call(arguments));
return _oExt.oApi[sFunc].apply(this,aArgs)}}for(var sFunc in _oExt.oApi){if(sFunc){this[sFunc]=_fnExternApiFunc(sFunc)
}}function _fnInitalise(oSettings){if(oSettings.bInitialised===false){setTimeout(function(){_fnInitalise(oSettings)
},200);return}_fnAddOptionsHtml(oSettings);_fnDrawHead(oSettings);if(oSettings.oFeatures.bSort){_fnSort(oSettings,false);
_fnSortingClasses(oSettings)}else{oSettings.aiDisplay=oSettings.aiDisplayMaster.slice();
_fnCalculateEnd(oSettings);_fnDraw(oSettings)}if(oSettings.sAjaxSource!==null&&!oSettings.oFeatures.bServerSide){_fnProcessingDisplay(oSettings,true);
oSettings.fnServerData(oSettings.sAjaxSource,null,function(json){for(var i=0;i<json.aaData.length;
i++){_fnAddData(oSettings,json.aaData[i])}oSettings.iInitDisplayStart=oSettings._iDisplayStart;
if(oSettings.oFeatures.bSort){_fnSort(oSettings)}else{oSettings.aiDisplay=oSettings.aiDisplayMaster.slice();
_fnCalculateEnd(oSettings);_fnDraw(oSettings)}_fnProcessingDisplay(oSettings,false);
if(typeof oSettings.fnInitComplete=="function"){oSettings.fnInitComplete(oSettings,json)
}});return}if(typeof oSettings.fnInitComplete=="function"){oSettings.fnInitComplete(oSettings)
}if(!oSettings.oFeatures.bServerSide){_fnProcessingDisplay(oSettings,false)}}function _fnLanguageProcess(oSettings,oLanguage,bInit){_fnMap(oSettings.oLanguage,oLanguage,"sProcessing");
_fnMap(oSettings.oLanguage,oLanguage,"sLengthMenu");_fnMap(oSettings.oLanguage,oLanguage,"sZeroRecords");
_fnMap(oSettings.oLanguage,oLanguage,"sInfo");_fnMap(oSettings.oLanguage,oLanguage,"sInfoEmpty");
_fnMap(oSettings.oLanguage,oLanguage,"sInfoFiltered");_fnMap(oSettings.oLanguage,oLanguage,"sInfoPostFix");
_fnMap(oSettings.oLanguage,oLanguage,"sSearch");if(typeof oLanguage.oPaginate!="undefined"){_fnMap(oSettings.oLanguage.oPaginate,oLanguage.oPaginate,"sFirst");
_fnMap(oSettings.oLanguage.oPaginate,oLanguage.oPaginate,"sPrevious");_fnMap(oSettings.oLanguage.oPaginate,oLanguage.oPaginate,"sNext");
_fnMap(oSettings.oLanguage.oPaginate,oLanguage.oPaginate,"sLast")}if(bInit){_fnInitalise(oSettings)
}}function _fnAddColumn(oSettings,oOptions,nTh){oSettings.aoColumns[oSettings.aoColumns.length++]={sType:null,_bAutoType:true,bVisible:true,bSearchable:true,bSortable:true,asSorting:["asc","desc"],sSortingClass:oSettings.oClasses.sSortable,sSortingClassJUI:oSettings.oClasses.sSortJUI,sTitle:nTh?nTh.innerHTML:"",sName:"",sWidth:null,sClass:null,fnRender:null,bUseRendered:true,iDataSort:oSettings.aoColumns.length-1,sSortDataType:"std",nTh:nTh?nTh:document.createElement("th"),nTf:null};
var iLength=oSettings.aoColumns.length-1;var oCol=oSettings.aoColumns[iLength];if(typeof oOptions!="undefined"&&oOptions!==null){if(typeof oOptions.sType!="undefined"){oCol.sType=oOptions.sType;
oCol._bAutoType=false}_fnMap(oCol,oOptions,"bVisible");_fnMap(oCol,oOptions,"bSearchable");
_fnMap(oCol,oOptions,"bSortable");_fnMap(oCol,oOptions,"sTitle");_fnMap(oCol,oOptions,"sName");
_fnMap(oCol,oOptions,"sWidth");_fnMap(oCol,oOptions,"sClass");_fnMap(oCol,oOptions,"fnRender");
_fnMap(oCol,oOptions,"bUseRendered");_fnMap(oCol,oOptions,"iDataSort");_fnMap(oCol,oOptions,"asSorting");
_fnMap(oCol,oOptions,"sSortDataType")}if(!oSettings.oFeatures.bSort){oCol.bSortable=false
}if(!oCol.bSortable||($.inArray("asc",oCol.asSorting)==-1&&$.inArray("desc",oCol.asSorting)==-1)){oCol.sSortingClass=oSettings.oClasses.sSortableNone;
oCol.sSortingClassJUI=""}else{if($.inArray("asc",oCol.asSorting)!=-1&&$.inArray("desc",oCol.asSorting)==-1){oCol.sSortingClass=oSettings.oClasses.sSortableAsc;
oCol.sSortingClassJUI=oSettings.oClasses.sSortJUIAscAllowed}else{if($.inArray("asc",oCol.asSorting)==-1&&$.inArray("desc",oCol.asSorting)!=-1){oCol.sSortingClass=oSettings.oClasses.sSortableDesc;
oCol.sSortingClassJUI=oSettings.oClasses.sSortJUIDescAllowed}}}if(typeof oSettings.aoPreSearchCols[iLength]=="undefined"||oSettings.aoPreSearchCols[iLength]===null){oSettings.aoPreSearchCols[iLength]={sSearch:"",bEscapeRegex:true}
}else{if(typeof oSettings.aoPreSearchCols[iLength].bEscapeRegex=="undefined"){oSettings.aoPreSearchCols[iLength].bEscapeRegex=true
}}}function _fnAddData(oSettings,aData){if(aData.length!=oSettings.aoColumns.length){alert("DataTables warning: Added data does not match known number of columns");
return -1}var iThisIndex=oSettings.aoData.length;oSettings.aoData.push({nTr:document.createElement("tr"),_iId:oSettings.iNextId++,_aData:aData.slice(),_anHidden:[],_sRowStripe:""});
var nTd,sThisType;for(var i=0;i<aData.length;i++){nTd=document.createElement("td");
if(typeof oSettings.aoColumns[i].fnRender=="function"){var sRendered=oSettings.aoColumns[i].fnRender({iDataRow:iThisIndex,iDataColumn:i,aData:aData,oSettings:oSettings});
nTd.innerHTML=sRendered;if(oSettings.aoColumns[i].bUseRendered){oSettings.aoData[iThisIndex]._aData[i]=sRendered
}}else{nTd.innerHTML=aData[i]}if(oSettings.aoColumns[i].sClass!==null){nTd.className=oSettings.aoColumns[i].sClass
}if(oSettings.aoColumns[i]._bAutoType&&oSettings.aoColumns[i].sType!="string"){sThisType=_fnDetectType(oSettings.aoData[iThisIndex]._aData[i]);
if(oSettings.aoColumns[i].sType===null){oSettings.aoColumns[i].sType=sThisType}else{if(oSettings.aoColumns[i].sType!=sThisType){oSettings.aoColumns[i].sType="string"
}}}if(oSettings.aoColumns[i].bVisible){oSettings.aoData[iThisIndex].nTr.appendChild(nTd)
}else{oSettings.aoData[iThisIndex]._anHidden[i]=nTd}}oSettings.aiDisplayMaster.push(iThisIndex);
return iThisIndex}function _fnGatherData(oSettings){var iLoop,i,iLen,j,jLen,jInner,nTds,nTrs,nTd,aLocalData,iThisIndex,iRow,iRows,iColumn,iColumns;
if(oSettings.sAjaxSource===null){nTrs=oSettings.nTable.getElementsByTagName("tbody")[0].childNodes;
for(i=0,iLen=nTrs.length;i<iLen;i++){if(nTrs[i].nodeName=="TR"){iThisIndex=oSettings.aoData.length;
oSettings.aoData.push({nTr:nTrs[i],_iId:oSettings.iNextId++,_aData:[],_anHidden:[],_sRowStripe:""});
oSettings.aiDisplayMaster.push(iThisIndex);aLocalData=oSettings.aoData[iThisIndex]._aData;
nTds=nTrs[i].childNodes;jInner=0;for(j=0,jLen=nTds.length;j<jLen;j++){if(nTds[j].nodeName=="TD"){aLocalData[jInner]=nTds[j].innerHTML;
jInner++}}}}}nTrs=_fnGetTrNodes(oSettings);nTds=[];for(i=0,iLen=nTrs.length;i<iLen;
i++){for(j=0,jLen=nTrs[i].childNodes.length;j<jLen;j++){nTd=nTrs[i].childNodes[j];
if(nTd.nodeName=="TD"){nTds.push(nTd)}}}if(nTds.length!=nTrs.length*oSettings.aoColumns.length){alert("DataTables warning: Unexpected number of TD elements. Expected "+(nTrs.length*oSettings.aoColumns.length)+" and got "+nTds.length+". DataTables does not support rowspan / colspan in the table body, and there must be one cell for each row/column combination.")
}for(iColumn=0,iColumns=oSettings.aoColumns.length;iColumn<iColumns;iColumn++){if(oSettings.aoColumns[iColumn].sTitle===null){oSettings.aoColumns[iColumn].sTitle=oSettings.aoColumns[iColumn].nTh.innerHTML
}var bAutoType=oSettings.aoColumns[iColumn]._bAutoType,bRender=typeof oSettings.aoColumns[iColumn].fnRender=="function",bClass=oSettings.aoColumns[iColumn].sClass!==null,bVisible=oSettings.aoColumns[iColumn].bVisible,nCell,sThisType,sRendered;
if(bAutoType||bRender||bClass||!bVisible){for(iRow=0,iRows=oSettings.aoData.length;
iRow<iRows;iRow++){nCell=nTds[(iRow*iColumns)+iColumn];if(bAutoType){if(oSettings.aoColumns[iColumn].sType!="string"){sThisType=_fnDetectType(oSettings.aoData[iRow]._aData[iColumn]);
if(oSettings.aoColumns[iColumn].sType===null){oSettings.aoColumns[iColumn].sType=sThisType
}else{if(oSettings.aoColumns[iColumn].sType!=sThisType){oSettings.aoColumns[iColumn].sType="string"
}}}}if(bRender){sRendered=oSettings.aoColumns[iColumn].fnRender({iDataRow:iRow,iDataColumn:iColumn,aData:oSettings.aoData[iRow]._aData,oSettings:oSettings});
nCell.innerHTML=sRendered;if(oSettings.aoColumns[iColumn].bUseRendered){oSettings.aoData[iRow]._aData[iColumn]=sRendered
}}if(bClass){nCell.className+=" "+oSettings.aoColumns[iColumn].sClass}if(!bVisible){oSettings.aoData[iRow]._anHidden[iColumn]=nCell;
nCell.parentNode.removeChild(nCell)}}}}}function _fnDrawHead(oSettings){var i,nTh,iLen;
var iThs=oSettings.nTable.getElementsByTagName("thead")[0].getElementsByTagName("th").length;
var iCorrector=0;if(iThs!==0){for(i=0,iLen=oSettings.aoColumns.length;i<iLen;i++){nTh=oSettings.aoColumns[i].nTh;
if(oSettings.aoColumns[i].bVisible){if(oSettings.aoColumns[i].sWidth!==null){nTh.style.width=oSettings.aoColumns[i].sWidth
}if(oSettings.aoColumns[i].sTitle!=nTh.innerHTML){nTh.innerHTML=oSettings.aoColumns[i].sTitle
}}else{nTh.parentNode.removeChild(nTh);iCorrector++}}}else{var nTr=document.createElement("tr");
for(i=0,iLen=oSettings.aoColumns.length;i<iLen;i++){nTh=oSettings.aoColumns[i].nTh;
nTh.innerHTML=oSettings.aoColumns[i].sTitle;if(oSettings.aoColumns[i].bVisible){if(oSettings.aoColumns[i].sClass!==null){nTh.className=oSettings.aoColumns[i].sClass
}if(oSettings.aoColumns[i].sWidth!==null){nTh.style.width=oSettings.aoColumns[i].sWidth
}nTr.appendChild(nTh)}}$("thead:eq(0)",oSettings.nTable).html("")[0].appendChild(nTr)
}if(oSettings.bJUI){for(i=0,iLen=oSettings.aoColumns.length;i<iLen;i++){oSettings.aoColumns[i].nTh.insertBefore(document.createElement("span"),oSettings.aoColumns[i].nTh.firstChild)
}}if(oSettings.oFeatures.bSort){for(i=0;i<oSettings.aoColumns.length;i++){if(oSettings.aoColumns[i].bSortable!==false){_fnSortAttachListener(oSettings,oSettings.aoColumns[i].nTh,i)
}else{$(oSettings.aoColumns[i].nTh).addClass(oSettings.oClasses.sSortableNone)}}$("thead:eq(0) th",oSettings.nTable).mousedown(function(e){if(e.shiftKey){this.onselectstart=function(){return false
};return false}})}var nTfoot=oSettings.nTable.getElementsByTagName("tfoot");if(nTfoot.length!==0){iCorrector=0;
var nTfs=nTfoot[0].getElementsByTagName("th");for(i=0,iLen=nTfs.length;i<iLen;i++){oSettings.aoColumns[i].nTf=nTfs[i-iCorrector];
if(!oSettings.aoColumns[i].bVisible){nTfs[i-iCorrector].parentNode.removeChild(nTfs[i-iCorrector]);
iCorrector++}}}}function _fnDraw(oSettings){var i,iLen;var anRows=[];var iRowCount=0;
var bRowError=false;var iStrips=oSettings.asStripClasses.length;var iOpenRows=oSettings.aoOpenRows.length;
if(oSettings.oFeatures.bServerSide&&!_fnAjaxUpdate(oSettings)){return}if(typeof oSettings.iInitDisplayStart!="undefined"&&oSettings.iInitDisplayStart!=-1){oSettings._iDisplayStart=(oSettings.iInitDisplayStart>=oSettings.fnRecordsDisplay())?0:oSettings.iInitDisplayStart;
oSettings.iInitDisplayStart=-1;_fnCalculateEnd(oSettings)}if(oSettings.aiDisplay.length!==0){var iStart=oSettings._iDisplayStart;
var iEnd=oSettings._iDisplayEnd;if(oSettings.oFeatures.bServerSide){iStart=0;iEnd=oSettings.aoData.length
}for(var j=iStart;j<iEnd;j++){var aoData=oSettings.aoData[oSettings.aiDisplay[j]];
var nRow=aoData.nTr;if(iStrips!==0){var sStrip=oSettings.asStripClasses[iRowCount%iStrips];
if(aoData._sRowStripe!=sStrip){$(nRow).removeClass(aoData._sRowStripe).addClass(sStrip);
aoData._sRowStripe=sStrip}}if(typeof oSettings.fnRowCallback=="function"){nRow=oSettings.fnRowCallback(nRow,oSettings.aoData[oSettings.aiDisplay[j]]._aData,iRowCount,j);
if(!nRow&&!bRowError){alert("DataTables warning: A node was not returned by fnRowCallback");
bRowError=true}}anRows.push(nRow);iRowCount++;if(iOpenRows!==0){for(var k=0;k<iOpenRows;
k++){if(nRow==oSettings.aoOpenRows[k].nParent){anRows.push(oSettings.aoOpenRows[k].nTr)
}}}}}else{anRows[0]=document.createElement("tr");if(typeof oSettings.asStripClasses[0]!="undefined"){anRows[0].className=oSettings.asStripClasses[0]
}var nTd=document.createElement("td");nTd.setAttribute("valign","top");nTd.colSpan=oSettings.aoColumns.length;
nTd.className=oSettings.oClasses.sRowEmpty;nTd.innerHTML=oSettings.oLanguage.sZeroRecords;
anRows[iRowCount].appendChild(nTd)}if(typeof oSettings.fnHeaderCallback=="function"){oSettings.fnHeaderCallback($("thead:eq(0)>tr",oSettings.nTable)[0],_fnGetDataMaster(oSettings),oSettings._iDisplayStart,oSettings.fnDisplayEnd(),oSettings.aiDisplay)
}if(typeof oSettings.fnFooterCallback=="function"){oSettings.fnFooterCallback($("tfoot:eq(0)>tr",oSettings.nTable)[0],_fnGetDataMaster(oSettings),oSettings._iDisplayStart,oSettings.fnDisplayEnd(),oSettings.aiDisplay)
}var nBody=oSettings.nTable.getElementsByTagName("tbody");if(nBody[0]){var nTrs=nBody[0].childNodes;
for(i=nTrs.length-1;i>=0;i--){nTrs[i].parentNode.removeChild(nTrs[i])}for(i=0,iLen=anRows.length;
i<iLen;i++){nBody[0].appendChild(anRows[i])}}for(i=0,iLen=oSettings.aoDrawCallback.length;
i<iLen;i++){oSettings.aoDrawCallback[i].fn(oSettings)}oSettings.bSorted=false;oSettings.bFiltered=false;
if(typeof oSettings._bInitComplete=="undefined"){oSettings._bInitComplete=true;if(oSettings.oFeatures.bAutoWidth&&oSettings.nTable.offsetWidth!==0){oSettings.nTable.style.width=oSettings.nTable.offsetWidth+"px"
}}}function _fnReDraw(oSettings){if(oSettings.oFeatures.bSort){_fnSort(oSettings,oSettings.oPreviousSearch)
}else{if(oSettings.oFeatures.bFilter){_fnFilterComplete(oSettings,oSettings.oPreviousSearch)
}else{_fnCalculateEnd(oSettings);_fnDraw(oSettings)}}}function _fnAjaxUpdate(oSettings){if(oSettings.bAjaxDataGet){_fnProcessingDisplay(oSettings,true);
var iColumns=oSettings.aoColumns.length;var aoData=[];var i;oSettings.iServerDraw++;
aoData.push({name:"sEcho",value:oSettings.iServerDraw});aoData.push({name:"iColumns",value:iColumns});
aoData.push({name:"sColumns",value:_fnColumnOrdering(oSettings)});aoData.push({name:"iDisplayStart",value:oSettings._iDisplayStart});
aoData.push({name:"iDisplayLength",value:oSettings.oFeatures.bPaginate!==false?oSettings._iDisplayLength:-1});
if(oSettings.oFeatures.bFilter!==false){aoData.push({name:"sSearch",value:oSettings.oPreviousSearch.sSearch});
aoData.push({name:"bEscapeRegex",value:oSettings.oPreviousSearch.bEscapeRegex});for(i=0;
i<iColumns;i++){aoData.push({name:"sSearch_"+i,value:oSettings.aoPreSearchCols[i].sSearch});
aoData.push({name:"bEscapeRegex_"+i,value:oSettings.aoPreSearchCols[i].bEscapeRegex});
aoData.push({name:"bSearchable_"+i,value:oSettings.aoColumns[i].bSearchable})}}if(oSettings.oFeatures.bSort!==false){var iFixed=oSettings.aaSortingFixed!==null?oSettings.aaSortingFixed.length:0;
var iUser=oSettings.aaSorting.length;aoData.push({name:"iSortingCols",value:iFixed+iUser});
for(i=0;i<iFixed;i++){aoData.push({name:"iSortCol_"+i,value:oSettings.aaSortingFixed[i][0]});
aoData.push({name:"sSortDir_"+i,value:oSettings.aaSortingFixed[i][1]})}for(i=0;i<iUser;
i++){aoData.push({name:"iSortCol_"+(i+iFixed),value:oSettings.aaSorting[i][0]});aoData.push({name:"sSortDir_"+(i+iFixed),value:oSettings.aaSorting[i][1]})
}for(i=0;i<iColumns;i++){aoData.push({name:"bSortable_"+i,value:oSettings.aoColumns[i].bSortable})
}}oSettings.fnServerData(oSettings.sAjaxSource,aoData,function(json){_fnAjaxUpdateDraw(oSettings,json)
});return false}else{return true}}function _fnAjaxUpdateDraw(oSettings,json){if(typeof json.sEcho!="undefined"){if(json.sEcho*1<oSettings.iServerDraw){return
}else{oSettings.iServerDraw=json.sEcho*1}}_fnClearTable(oSettings);oSettings._iRecordsTotal=json.iTotalRecords;
oSettings._iRecordsDisplay=json.iTotalDisplayRecords;var sOrdering=_fnColumnOrdering(oSettings);
var bReOrder=(typeof json.sColumns!="undefined"&&sOrdering!==""&&json.sColumns!=sOrdering);
if(bReOrder){var aiIndex=_fnReOrderIndex(oSettings,json.sColumns)}for(var i=0,iLen=json.aaData.length;
i<iLen;i++){if(bReOrder){var aData=[];for(var j=0,jLen=oSettings.aoColumns.length;
j<jLen;j++){aData.push(json.aaData[i][aiIndex[j]])}_fnAddData(oSettings,aData)}else{_fnAddData(oSettings,json.aaData[i])
}}oSettings.aiDisplay=oSettings.aiDisplayMaster.slice();oSettings.bAjaxDataGet=false;
_fnDraw(oSettings);oSettings.bAjaxDataGet=true;_fnProcessingDisplay(oSettings,false)
}function _fnAddOptionsHtml(oSettings){var nHolding=document.createElement("div");
oSettings.nTable.parentNode.insertBefore(nHolding,oSettings.nTable);var nWrapper=document.createElement("div");
nWrapper.className=oSettings.oClasses.sWrapper;if(oSettings.sTableId!==""){nWrapper.setAttribute("id",oSettings.sTableId+"_wrapper")
}var nInsertNode=nWrapper;var sDom=oSettings.sDom.replace("H","fg-toolbar ui-widget-header ui-corner-tl ui-corner-tr ui-helper-clearfix");
sDom=sDom.replace("F","fg-toolbar ui-widget-header ui-corner-bl ui-corner-br ui-helper-clearfix");
var aDom=sDom.split("");var nTmp,iPushFeature,cOption,nNewNode,cNext,sClass,j;for(var i=0;
i<aDom.length;i++){iPushFeature=0;cOption=aDom[i];if(cOption=="<"){nNewNode=document.createElement("div");
cNext=aDom[i+1];if(cNext=="'"||cNext=='"'){sClass="";j=2;while(aDom[i+j]!=cNext){sClass+=aDom[i+j];
j++}nNewNode.className=sClass;i+=j}nInsertNode.appendChild(nNewNode);nInsertNode=nNewNode
}else{if(cOption==">"){nInsertNode=nInsertNode.parentNode}else{if(cOption=="l"&&oSettings.oFeatures.bPaginate&&oSettings.oFeatures.bLengthChange){nTmp=_fnFeatureHtmlLength(oSettings);
iPushFeature=1}else{if(cOption=="f"&&oSettings.oFeatures.bFilter){nTmp=_fnFeatureHtmlFilter(oSettings);
iPushFeature=1}else{if(cOption=="r"&&oSettings.oFeatures.bProcessing){nTmp=_fnFeatureHtmlProcessing(oSettings);
iPushFeature=1}else{if(cOption=="t"){nTmp=oSettings.nTable;iPushFeature=1}else{if(cOption=="i"&&oSettings.oFeatures.bInfo){nTmp=_fnFeatureHtmlInfo(oSettings);
iPushFeature=1}else{if(cOption=="p"&&oSettings.oFeatures.bPaginate){nTmp=_fnFeatureHtmlPaginate(oSettings);
iPushFeature=1}else{if(_oExt.aoFeatures.length!==0){var aoFeatures=_oExt.aoFeatures;
for(var k=0,kLen=aoFeatures.length;k<kLen;k++){if(cOption==aoFeatures[k].cFeature){nTmp=aoFeatures[k].fnInit(oSettings);
if(nTmp){iPushFeature=1}break}}}}}}}}}}}if(iPushFeature==1){if(typeof oSettings.aanFeatures[cOption]!="object"){oSettings.aanFeatures[cOption]=[]
}oSettings.aanFeatures[cOption].push(nTmp);nInsertNode.appendChild(nTmp)}}nHolding.parentNode.replaceChild(nWrapper,nHolding)
}function _fnFeatureHtmlFilter(oSettings){var nFilter=document.createElement("div");
if(oSettings.sTableId!==""&&typeof oSettings.aanFeatures.f=="undefined"){nFilter.setAttribute("id",oSettings.sTableId+"_filter")
}nFilter.className=oSettings.oClasses.sFilter;var sSpace=oSettings.oLanguage.sSearch===""?"":" ";
nFilter.innerHTML=oSettings.oLanguage.sSearch+sSpace+'<input type="text" />';var jqFilter=$("input",nFilter);
jqFilter.val(oSettings.oPreviousSearch.sSearch.replace('"',"&quot;"));jqFilter.keyup(function(e){var n=oSettings.aanFeatures.f;
for(var i=0,iLen=n.length;i<iLen;i++){if(n[i]!=this.parentNode){$("input",n[i]).val(this.value)
}}_fnFilterComplete(oSettings,{sSearch:this.value,bEscapeRegex:oSettings.oPreviousSearch.bEscapeRegex})
});jqFilter.keypress(function(e){if(e.keyCode==13){return false}});return nFilter
}function _fnFilterComplete(oSettings,oInput,iForce){_fnFilter(oSettings,oInput.sSearch,iForce,oInput.bEscapeRegex);
for(var i=0;i<oSettings.aoPreSearchCols.length;i++){_fnFilterColumn(oSettings,oSettings.aoPreSearchCols[i].sSearch,i,oSettings.aoPreSearchCols[i].bEscapeRegex)
}if(_oExt.afnFiltering.length!==0){_fnFilterCustom(oSettings)}oSettings.bFiltered=true;
oSettings._iDisplayStart=0;_fnCalculateEnd(oSettings);_fnDraw(oSettings);_fnBuildSearchArray(oSettings,0)
}function _fnFilterCustom(oSettings){var afnFilters=_oExt.afnFiltering;for(var i=0,iLen=afnFilters.length;
i<iLen;i++){var iCorrector=0;for(var j=0,jLen=oSettings.aiDisplay.length;j<jLen;j++){var iDisIndex=oSettings.aiDisplay[j-iCorrector];
if(!afnFilters[i](oSettings,oSettings.aoData[iDisIndex]._aData,iDisIndex)){oSettings.aiDisplay.splice(j-iCorrector,1);
iCorrector++}}}}function _fnFilterColumn(oSettings,sInput,iColumn,bEscapeRegex){if(sInput===""){return
}var iIndexCorrector=0;var sRegexMatch=bEscapeRegex?_fnEscapeRegex(sInput):sInput;
var rpSearch=new RegExp(sRegexMatch,"i");for(var i=oSettings.aiDisplay.length-1;i>=0;
i--){var sData=_fnDataToSearch(oSettings.aoData[oSettings.aiDisplay[i]]._aData[iColumn],oSettings.aoColumns[iColumn].sType);
if(!rpSearch.test(sData)){oSettings.aiDisplay.splice(i,1);iIndexCorrector++}}}function _fnFilter(oSettings,sInput,iForce,bEscapeRegex){var i;
if(typeof iForce=="undefined"||iForce===null){iForce=0}if(_oExt.afnFiltering.length!==0){iForce=1
}var asSearch=bEscapeRegex?_fnEscapeRegex(sInput).split(" "):sInput.split(" ");var sRegExpString="^(?=.*?"+asSearch.join(")(?=.*?")+").*$";
var rpSearch=new RegExp(sRegExpString,"i");if(sInput.length<=0){oSettings.aiDisplay.splice(0,oSettings.aiDisplay.length);
oSettings.aiDisplay=oSettings.aiDisplayMaster.slice()}else{if(oSettings.aiDisplay.length==oSettings.aiDisplayMaster.length||oSettings.oPreviousSearch.sSearch.length>sInput.length||iForce==1||sInput.indexOf(oSettings.oPreviousSearch.sSearch)!==0){oSettings.aiDisplay.splice(0,oSettings.aiDisplay.length);
_fnBuildSearchArray(oSettings,1);for(i=0;i<oSettings.aiDisplayMaster.length;i++){if(rpSearch.test(oSettings.asDataSearch[i])){oSettings.aiDisplay.push(oSettings.aiDisplayMaster[i])
}}}else{var iIndexCorrector=0;for(i=0;i<oSettings.asDataSearch.length;i++){if(!rpSearch.test(oSettings.asDataSearch[i])){oSettings.aiDisplay.splice(i-iIndexCorrector,1);
iIndexCorrector++}}}}oSettings.oPreviousSearch.sSearch=sInput;oSettings.oPreviousSearch.bEscapeRegex=bEscapeRegex
}function _fnBuildSearchArray(oSettings,iMaster){oSettings.asDataSearch.splice(0,oSettings.asDataSearch.length);
var aArray=(typeof iMaster!="undefined"&&iMaster==1)?oSettings.aiDisplayMaster:oSettings.aiDisplay;
for(var i=0,iLen=aArray.length;i<iLen;i++){oSettings.asDataSearch[i]="";for(var j=0,jLen=oSettings.aoColumns.length;
j<jLen;j++){if(oSettings.aoColumns[j].bSearchable){var sData=oSettings.aoData[aArray[i]]._aData[j];
oSettings.asDataSearch[i]+=_fnDataToSearch(sData,oSettings.aoColumns[j].sType)+" "
}}}}function _fnDataToSearch(sData,sType){if(typeof _oExt.ofnSearch[sType]=="function"){return _oExt.ofnSearch[sType](sData)
}else{if(sType=="html"){return sData.replace(/\n/g," ").replace(/<.*?>/g,"")}else{if(typeof sData=="string"){return sData.replace(/\n/g," ")
}}}return sData}function _fnSort(oSettings,bApplyClasses){var aaSort=[];var oSort=_oExt.oSort;
var aoData=oSettings.aoData;var iDataSort;var iDataType;var i,j,jLen;if(!oSettings.oFeatures.bServerSide&&(oSettings.aaSorting.length!==0||oSettings.aaSortingFixed!==null)){if(oSettings.aaSortingFixed!==null){aaSort=oSettings.aaSortingFixed.concat(oSettings.aaSorting)
}else{aaSort=oSettings.aaSorting.slice()}for(i=0;i<aaSort.length;i++){var iColumn=aaSort[i][0];
var sDataType=oSettings.aoColumns[iColumn].sSortDataType;if(typeof _oExt.afnSortData[sDataType]!="undefined"){var iCorrector=0;
var aData=_oExt.afnSortData[sDataType](oSettings,iColumn);for(j=0,jLen=aoData.length;
j<jLen;j++){if(aoData[j]!==null){aoData[j]._aData[iColumn]=aData[iCorrector];iCorrector++
}}}}if(!window.runtime){var fnLocalSorting;var sDynamicSort="fnLocalSorting = function(a,b){var iTest;";
for(i=0;i<aaSort.length-1;i++){iDataSort=oSettings.aoColumns[aaSort[i][0]].iDataSort;
iDataType=oSettings.aoColumns[iDataSort].sType;sDynamicSort+="iTest = oSort['"+iDataType+"-"+aaSort[i][1]+"']( aoData[a]._aData["+iDataSort+"], aoData[b]._aData["+iDataSort+"] ); if ( iTest === 0 )"
}iDataSort=oSettings.aoColumns[aaSort[aaSort.length-1][0]].iDataSort;iDataType=oSettings.aoColumns[iDataSort].sType;
sDynamicSort+="iTest = oSort['"+iDataType+"-"+aaSort[aaSort.length-1][1]+"']( aoData[a]._aData["+iDataSort+"], aoData[b]._aData["+iDataSort+"] );if (iTest===0) return oSort['numeric-"+aaSort[aaSort.length-1][1]+"'](a, b); return iTest;}";
eval(sDynamicSort);oSettings.aiDisplayMaster.sort(fnLocalSorting)}else{var aAirSort=[];
var iLen=aaSort.length;for(i=0;i<iLen;i++){iDataSort=oSettings.aoColumns[aaSort[i][0]].iDataSort;
aAirSort.push([iDataSort,oSettings.aoColumns[iDataSort].sType+"-"+aaSort[i][1]])}oSettings.aiDisplayMaster.sort(function(a,b){var iTest;
for(var i=0;i<iLen;i++){iTest=oSort[aAirSort[i][1]](aoData[a]._aData[aAirSort[i][0]],aoData[b]._aData[aAirSort[i][0]]);
if(iTest!==0){return iTest}}return 0})}}if(typeof bApplyClasses=="undefined"||bApplyClasses){_fnSortingClasses(oSettings)
}oSettings.bSorted=true;if(oSettings.oFeatures.bFilter){_fnFilterComplete(oSettings,oSettings.oPreviousSearch,1)
}else{oSettings.aiDisplay=oSettings.aiDisplayMaster.slice();oSettings._iDisplayStart=0;
_fnCalculateEnd(oSettings);_fnDraw(oSettings)}}function _fnSortAttachListener(oSettings,nNode,iDataIndex,fnCallback){$(nNode).click(function(e){if(oSettings.aoColumns[iDataIndex].bSortable===false){return
}var fnInnerSorting=function(){var iColumn,iNextSort;if(e.shiftKey){var bFound=false;
for(var i=0;i<oSettings.aaSorting.length;i++){if(oSettings.aaSorting[i][0]==iDataIndex){bFound=true;
iColumn=oSettings.aaSorting[i][0];iNextSort=oSettings.aaSorting[i][2]+1;if(typeof oSettings.aoColumns[iColumn].asSorting[iNextSort]=="undefined"){oSettings.aaSorting.splice(i,1)
}else{oSettings.aaSorting[i][1]=oSettings.aoColumns[iColumn].asSorting[iNextSort];
oSettings.aaSorting[i][2]=iNextSort}break}}if(bFound===false){oSettings.aaSorting.push([iDataIndex,oSettings.aoColumns[iDataIndex].asSorting[0],0])
}}else{if(oSettings.aaSorting.length==1&&oSettings.aaSorting[0][0]==iDataIndex){iColumn=oSettings.aaSorting[0][0];
iNextSort=oSettings.aaSorting[0][2]+1;if(typeof oSettings.aoColumns[iColumn].asSorting[iNextSort]=="undefined"){iNextSort=0
}oSettings.aaSorting[0][1]=oSettings.aoColumns[iColumn].asSorting[iNextSort];oSettings.aaSorting[0][2]=iNextSort
}else{oSettings.aaSorting.splice(0,oSettings.aaSorting.length);oSettings.aaSorting.push([iDataIndex,oSettings.aoColumns[iDataIndex].asSorting[0],0])
}}_fnSort(oSettings)};if(!oSettings.oFeatures.bProcessing){fnInnerSorting()}else{_fnProcessingDisplay(oSettings,true);
setTimeout(function(){fnInnerSorting();if(!oSettings.oFeatures.bServerSide){_fnProcessingDisplay(oSettings,false)
}},0)}if(typeof fnCallback=="function"){fnCallback(oSettings)}})}function _fnSortingClasses(oSettings){var i,iLen,j,jLen,iFound;
var aaSort,sClass;var iColumns=oSettings.aoColumns.length;var oClasses=oSettings.oClasses;
for(i=0;i<iColumns;i++){if(oSettings.aoColumns[i].bSortable){$(oSettings.aoColumns[i].nTh).removeClass(oClasses.sSortAsc+" "+oClasses.sSortDesc+" "+oSettings.aoColumns[i].sSortingClass)
}}if(oSettings.aaSortingFixed!==null){aaSort=oSettings.aaSortingFixed.concat(oSettings.aaSorting)
}else{aaSort=oSettings.aaSorting.slice()}for(i=0;i<oSettings.aoColumns.length;i++){if(oSettings.aoColumns[i].bSortable){sClass=oSettings.aoColumns[i].sSortingClass;
iFound=-1;for(j=0;j<aaSort.length;j++){if(aaSort[j][0]==i){sClass=(aaSort[j][1]=="asc")?oClasses.sSortAsc:oClasses.sSortDesc;
iFound=j;break}}$(oSettings.aoColumns[i].nTh).addClass(sClass);if(oSettings.bJUI){var jqSpan=$("span",oSettings.aoColumns[i].nTh);
jqSpan.removeClass(oClasses.sSortJUIAsc+" "+oClasses.sSortJUIDesc+" "+oClasses.sSortJUI+" "+oClasses.sSortJUIAscAllowed+" "+oClasses.sSortJUIDescAllowed);
var sSpanClass;if(iFound==-1){sSpanClass=oSettings.aoColumns[i].sSortingClassJUI}else{if(aaSort[iFound][1]=="asc"){sSpanClass=oClasses.sSortJUIAsc
}else{sSpanClass=oClasses.sSortJUIDesc}}jqSpan.addClass(sSpanClass)}}else{$(oSettings.aoColumns[i].nTh).addClass(oSettings.aoColumns[i].sSortingClass)
}}sClass=oClasses.sSortColumn;if(oSettings.oFeatures.bSort&&oSettings.oFeatures.bSortClasses){var nTds=_fnGetTdNodes(oSettings);
if(nTds.length>=iColumns){for(i=0;i<iColumns;i++){if(nTds[i].className.indexOf(sClass+"1")!=-1){for(j=0,jLen=(nTds.length/iColumns);
j<jLen;j++){nTds[(iColumns*j)+i].className=nTds[(iColumns*j)+i].className.replace(" "+sClass+"1","")
}}else{if(nTds[i].className.indexOf(sClass+"2")!=-1){for(j=0,jLen=(nTds.length/iColumns);
j<jLen;j++){nTds[(iColumns*j)+i].className=nTds[(iColumns*j)+i].className.replace(" "+sClass+"2","")
}}else{if(nTds[i].className.indexOf(sClass+"3")!=-1){for(j=0,jLen=(nTds.length/iColumns);
j<jLen;j++){nTds[(iColumns*j)+i].className=nTds[(iColumns*j)+i].className.replace(" "+sClass+"3","")
}}}}}}var iClass=1,iTargetCol;for(i=0;i<aaSort.length;i++){iTargetCol=parseInt(aaSort[i][0],10);
for(j=0,jLen=(nTds.length/iColumns);j<jLen;j++){nTds[(iColumns*j)+iTargetCol].className+=" "+sClass+iClass
}if(iClass<3){iClass++}}}}function _fnFeatureHtmlPaginate(oSettings){var nPaginate=document.createElement("div");
nPaginate.className=oSettings.oClasses.sPaging+oSettings.sPaginationType;_oExt.oPagination[oSettings.sPaginationType].fnInit(oSettings,nPaginate,function(oSettings){_fnCalculateEnd(oSettings);
_fnDraw(oSettings)});if(typeof oSettings.aanFeatures.p=="undefined"){oSettings.aoDrawCallback.push({fn:function(oSettings){_oExt.oPagination[oSettings.sPaginationType].fnUpdate(oSettings,function(oSettings){_fnCalculateEnd(oSettings);
_fnDraw(oSettings)})},sName:"pagination"})}return nPaginate}function _fnPageChange(oSettings,sAction){var iOldStart=oSettings._iDisplayStart;
if(sAction=="first"){oSettings._iDisplayStart=0}else{if(sAction=="previous"){oSettings._iDisplayStart=oSettings._iDisplayLength>=0?oSettings._iDisplayStart-oSettings._iDisplayLength:0;
if(oSettings._iDisplayStart<0){oSettings._iDisplayStart=0}}else{if(sAction=="next"){if(oSettings._iDisplayLength>=0){if(oSettings._iDisplayStart+oSettings._iDisplayLength<oSettings.fnRecordsDisplay()){oSettings._iDisplayStart+=oSettings._iDisplayLength
}}else{oSettings._iDisplayStart=0}}else{if(sAction=="last"){if(oSettings._iDisplayLength>=0){var iPages=parseInt((oSettings.fnRecordsDisplay()-1)/oSettings._iDisplayLength,10)+1;
oSettings._iDisplayStart=(iPages-1)*oSettings._iDisplayLength}else{oSettings._iDisplayStart=0
}}else{alert("DataTables warning: unknown paging action: "+sAction)}}}}return iOldStart!=oSettings._iDisplayStart
}function _fnFeatureHtmlInfo(oSettings){var nInfo=document.createElement("div");nInfo.className=oSettings.oClasses.sInfo;
if(typeof oSettings.aanFeatures.i=="undefined"){oSettings.aoDrawCallback.push({fn:_fnUpdateInfo,sName:"information"});
if(oSettings.sTableId!==""){nInfo.setAttribute("id",oSettings.sTableId+"_info")}}return nInfo
}function _fnUpdateInfo(oSettings){if(!oSettings.oFeatures.bInfo||oSettings.aanFeatures.i.length===0){return
}var nFirst=oSettings.aanFeatures.i[0];if(oSettings.fnRecordsDisplay()===0&&oSettings.fnRecordsDisplay()==oSettings.fnRecordsTotal()){nFirst.innerHTML=oSettings.oLanguage.sInfoEmpty+oSettings.oLanguage.sInfoPostFix
}else{if(oSettings.fnRecordsDisplay()===0){nFirst.innerHTML=oSettings.oLanguage.sInfoEmpty+" "+oSettings.oLanguage.sInfoFiltered.replace("_MAX_",oSettings.fnRecordsTotal())+oSettings.oLanguage.sInfoPostFix
}else{if(oSettings.fnRecordsDisplay()==oSettings.fnRecordsTotal()){nFirst.innerHTML=oSettings.oLanguage.sInfo.replace("_START_",oSettings._iDisplayStart+1).replace("_END_",oSettings.fnDisplayEnd()).replace("_TOTAL_",oSettings.fnRecordsDisplay())+oSettings.oLanguage.sInfoPostFix
}else{nFirst.innerHTML=oSettings.oLanguage.sInfo.replace("_START_",oSettings._iDisplayStart+1).replace("_END_",oSettings.fnDisplayEnd()).replace("_TOTAL_",oSettings.fnRecordsDisplay())+" "+oSettings.oLanguage.sInfoFiltered.replace("_MAX_",oSettings.fnRecordsTotal())+oSettings.oLanguage.sInfoPostFix
}}}var n=oSettings.aanFeatures.i;if(n.length>1){var sInfo=nFirst.innerHTML;for(var i=1,iLen=n.length;
i<iLen;i++){n[i].innerHTML=sInfo}}}function _fnFeatureHtmlLength(oSettings){var sName=(oSettings.sTableId==="")?"":'name="'+oSettings.sTableId+'_length"';
var sStdMenu='<select size="1" '+sName+'><option value="10">10</option><option value="25">25</option><option value="50">50</option><option value="100">100</option></select>';
var nLength=document.createElement("div");if(oSettings.sTableId!==""&&typeof oSettings.aanFeatures.l=="undefined"){nLength.setAttribute("id",oSettings.sTableId+"_length")
}nLength.className=oSettings.oClasses.sLength;nLength.innerHTML=oSettings.oLanguage.sLengthMenu.replace("_MENU_",sStdMenu);
$('select option[value="'+oSettings._iDisplayLength+'"]',nLength).attr("selected",true);
$("select",nLength).change(function(e){var iVal=$(this).val();var n=oSettings.aanFeatures.l;
for(var i=0,iLen=n.length;i<iLen;i++){if(n[i]!=this.parentNode){$("select",n[i]).val(iVal)
}}oSettings._iDisplayLength=parseInt(iVal,10);_fnCalculateEnd(oSettings);if(oSettings._iDisplayEnd==oSettings.aiDisplay.length){oSettings._iDisplayStart=oSettings._iDisplayEnd-oSettings._iDisplayLength;
if(oSettings._iDisplayStart<0){oSettings._iDisplayStart=0}}if(oSettings._iDisplayLength==-1){oSettings._iDisplayStart=0
}_fnDraw(oSettings)});return nLength}function _fnFeatureHtmlProcessing(oSettings){var nProcessing=document.createElement("div");
if(oSettings.sTableId!==""&&typeof oSettings.aanFeatures.r=="undefined"){nProcessing.setAttribute("id",oSettings.sTableId+"_processing")
}nProcessing.innerHTML=oSettings.oLanguage.sProcessing;nProcessing.className=oSettings.oClasses.sProcessing;
oSettings.nTable.parentNode.insertBefore(nProcessing,oSettings.nTable);return nProcessing
}function _fnProcessingDisplay(oSettings,bShow){if(oSettings.oFeatures.bProcessing){var an=oSettings.aanFeatures.r;
for(var i=0,iLen=an.length;i<iLen;i++){an[i].style.visibility=bShow?"visible":"hidden"
}}}function _fnVisibleToColumnIndex(oSettings,iMatch){var iColumn=-1;for(var i=0;
i<oSettings.aoColumns.length;i++){if(oSettings.aoColumns[i].bVisible===true){iColumn++
}if(iColumn==iMatch){return i}}return null}function _fnColumnIndexToVisible(oSettings,iMatch){var iVisible=-1;
for(var i=0;i<oSettings.aoColumns.length;i++){if(oSettings.aoColumns[i].bVisible===true){iVisible++
}if(i==iMatch){return oSettings.aoColumns[i].bVisible===true?iVisible:null}}return null
}function _fnNodeToDataIndex(s,n){for(var i=0,iLen=s.aoData.length;i<iLen;i++){if(s.aoData[i]!==null&&s.aoData[i].nTr==n){return i
}}return null}function _fnVisbleColumns(oS){var iVis=0;for(var i=0;i<oS.aoColumns.length;
i++){if(oS.aoColumns[i].bVisible===true){iVis++}}return iVis}function _fnCalculateEnd(oSettings){if(oSettings.oFeatures.bPaginate===false){oSettings._iDisplayEnd=oSettings.aiDisplay.length
}else{if(oSettings._iDisplayStart+oSettings._iDisplayLength>oSettings.aiDisplay.length||oSettings._iDisplayLength==-1){oSettings._iDisplayEnd=oSettings.aiDisplay.length
}else{oSettings._iDisplayEnd=oSettings._iDisplayStart+oSettings._iDisplayLength}}}function _fnConvertToWidth(sWidth,nParent){if(!sWidth||sWidth===null||sWidth===""){return 0
}if(typeof nParent=="undefined"){nParent=document.getElementsByTagName("body")[0]
}var iWidth;var nTmp=document.createElement("div");nTmp.style.width=sWidth;nParent.appendChild(nTmp);
iWidth=nTmp.offsetWidth;nParent.removeChild(nTmp);return(iWidth)}function _fnCalculateColumnWidths(oSettings){var iTableWidth=oSettings.nTable.offsetWidth;
var iTotalUserIpSize=0;var iTmpWidth;var iVisibleColumns=0;var iColums=oSettings.aoColumns.length;
var i;var oHeaders=$("thead:eq(0)>th",oSettings.nTable);for(i=0;i<iColums;i++){if(oSettings.aoColumns[i].bVisible){iVisibleColumns++;
if(oSettings.aoColumns[i].sWidth!==null){iTmpWidth=_fnConvertToWidth(oSettings.aoColumns[i].sWidth,oSettings.nTable.parentNode);
iTotalUserIpSize+=iTmpWidth;oSettings.aoColumns[i].sWidth=iTmpWidth+"px"}}}if(iColums==oHeaders.length&&iTotalUserIpSize===0&&iVisibleColumns==iColums){for(i=0;
i<oSettings.aoColumns.length;i++){oSettings.aoColumns[i].sWidth=oHeaders[i].offsetWidth+"px"
}}else{var nCalcTmp=oSettings.nTable.cloneNode(false);nCalcTmp.setAttribute("id","");
var sTableTmp='<table class="'+nCalcTmp.className+'">';var sCalcHead="<tr>";var sCalcHtml="<tr>";
for(i=0;i<iColums;i++){if(oSettings.aoColumns[i].bVisible){sCalcHead+="<th>"+oSettings.aoColumns[i].sTitle+"</th>";
if(oSettings.aoColumns[i].sWidth!==null){var sWidth="";if(oSettings.aoColumns[i].sWidth!==null){sWidth=' style="width:'+oSettings.aoColumns[i].sWidth+';"'
}sCalcHtml+="<td"+sWidth+' tag_index="'+i+'">'+fnGetMaxLenString(oSettings,i)+"</td>"
}else{sCalcHtml+='<td tag_index="'+i+'">'+fnGetMaxLenString(oSettings,i)+"</td>"}}}sCalcHead+="</tr>";
sCalcHtml+="</tr>";nCalcTmp=$(sTableTmp+sCalcHead+sCalcHtml+"</table>")[0];nCalcTmp.style.width=iTableWidth+"px";
nCalcTmp.style.visibility="hidden";nCalcTmp.style.position="absolute";oSettings.nTable.parentNode.appendChild(nCalcTmp);
var oNodes=$("tr:eq(1)>td",nCalcTmp);var iIndex;for(i=0;i<oNodes.length;i++){iIndex=oNodes[i].getAttribute("tag_index");
var iContentWidth=$("td",nCalcTmp).eq(i).width();var iSetWidth=oSettings.aoColumns[i].sWidth?oSettings.aoColumns[i].sWidth.slice(0,-2):0;
oSettings.aoColumns[iIndex].sWidth=Math.max(iContentWidth,iSetWidth)+"px"}oSettings.nTable.parentNode.removeChild(nCalcTmp)
}}function fnGetMaxLenString(oSettings,iCol){var iMax=0;var iMaxIndex=-1;for(var i=0;
i<oSettings.aoData.length;i++){if(oSettings.aoData[i]._aData[iCol].length>iMax){iMax=oSettings.aoData[i]._aData[iCol].length;
iMaxIndex=i}}if(iMaxIndex>=0){return oSettings.aoData[iMaxIndex]._aData[iCol]}return""
}function _fnArrayCmp(aArray1,aArray2){if(aArray1.length!=aArray2.length){return 1
}for(var i=0;i<aArray1.length;i++){if(aArray1[i]!=aArray2[i]){return 2}}return 0}function _fnDetectType(sData){var aTypes=_oExt.aTypes;
var iLen=aTypes.length;for(var i=0;i<iLen;i++){var sType=aTypes[i](sData);if(sType!==null){return sType
}}return"string"}function _fnSettingsFromNode(nTable){for(var i=0;i<_aoSettings.length;
i++){if(_aoSettings[i].nTable==nTable){return _aoSettings[i]}}return null}function _fnGetDataMaster(oSettings){var aData=[];
var iLen=oSettings.aoData.length;for(var i=0;i<iLen;i++){if(oSettings.aoData[i]===null){aData.push(null)
}else{aData.push(oSettings.aoData[i]._aData)}}return aData}function _fnGetTrNodes(oSettings){var aNodes=[];
var iLen=oSettings.aoData.length;for(var i=0;i<iLen;i++){if(oSettings.aoData[i]===null){aNodes.push(null)
}else{aNodes.push(oSettings.aoData[i].nTr)}}return aNodes}function _fnGetTdNodes(oSettings){var nTrs=_fnGetTrNodes(oSettings);
var nTds=[],nTd;var anReturn=[];var iCorrector;var iRow,iRows,iColumn,iColumns;for(iRow=0,iRows=nTrs.length;
iRow<iRows;iRow++){nTds=[];for(iColumn=0,iColumns=nTrs[iRow].childNodes.length;iColumn<iColumns;
iColumn++){nTd=nTrs[iRow].childNodes[iColumn];if(nTd.nodeName=="TD"){nTds.push(nTd)
}}iCorrector=0;for(iColumn=0,iColumns=oSettings.aoColumns.length;iColumn<iColumns;
iColumn++){if(oSettings.aoColumns[iColumn].bVisible){anReturn.push(nTds[iColumn-iCorrector])
}else{anReturn.push(oSettings.aoData[iRow]._anHidden[iColumn]);iCorrector++}}}return anReturn
}function _fnEscapeRegex(sVal){var acEscape=["/",".","*","+","?","|","(",")","[","]","{","}","\\","$","^"];
var reReplace=new RegExp("(\\"+acEscape.join("|\\")+")","g");return sVal.replace(reReplace,"\\$1")
}function _fnReOrderIndex(oSettings,sColumns){var aColumns=sColumns.split(",");var aiReturn=[];
for(var i=0,iLen=oSettings.aoColumns.length;i<iLen;i++){for(var j=0;j<iLen;j++){if(oSettings.aoColumns[i].sName==aColumns[j]){aiReturn.push(j);
break}}}return aiReturn}function _fnColumnOrdering(oSettings){var sNames="";for(var i=0,iLen=oSettings.aoColumns.length;
i<iLen;i++){sNames+=oSettings.aoColumns[i].sName+","}if(sNames.length==iLen){return""
}return sNames.slice(0,-1)}function _fnClearTable(oSettings){oSettings.aoData.length=0;
oSettings.aiDisplayMaster.length=0;oSettings.aiDisplay.length=0;_fnCalculateEnd(oSettings)
}function _fnSaveState(oSettings){if(!oSettings.oFeatures.bStateSave){return}var i;
var sValue="{";sValue+='"iStart": '+oSettings._iDisplayStart+",";sValue+='"iEnd": '+oSettings._iDisplayEnd+",";
sValue+='"iLength": '+oSettings._iDisplayLength+",";sValue+='"sFilter": "'+oSettings.oPreviousSearch.sSearch.replace('"','\\"')+'",';
sValue+='"sFilterEsc": '+oSettings.oPreviousSearch.bEscapeRegex+",";sValue+='"aaSorting": [ ';
for(i=0;i<oSettings.aaSorting.length;i++){sValue+="["+oSettings.aaSorting[i][0]+",'"+oSettings.aaSorting[i][1]+"'],"
}sValue=sValue.substring(0,sValue.length-1);sValue+="],";sValue+='"aaSearchCols": [ ';
for(i=0;i<oSettings.aoPreSearchCols.length;i++){sValue+="['"+oSettings.aoPreSearchCols[i].sSearch.replace("'","'")+"',"+oSettings.aoPreSearchCols[i].bEscapeRegex+"],"
}sValue=sValue.substring(0,sValue.length-1);sValue+="],";sValue+='"abVisCols": [ ';
for(i=0;i<oSettings.aoColumns.length;i++){sValue+=oSettings.aoColumns[i].bVisible+","
}sValue=sValue.substring(0,sValue.length-1);sValue+="]";sValue+="}";_fnCreateCookie("SpryMedia_DataTables_"+oSettings.sInstance,sValue,oSettings.iCookieDuration)
}function _fnLoadState(oSettings,oInit){if(!oSettings.oFeatures.bStateSave){return
}var oData;var sData=_fnReadCookie("SpryMedia_DataTables_"+oSettings.sInstance);if(sData!==null&&sData!==""){try{if(typeof JSON=="object"&&typeof JSON.parse=="function"){oData=JSON.parse(sData.replace(/'/g,'"'))
}else{oData=eval("("+sData+")")}}catch(e){return}oSettings._iDisplayStart=oData.iStart;
oSettings.iInitDisplayStart=oData.iStart;oSettings._iDisplayEnd=oData.iEnd;oSettings._iDisplayLength=oData.iLength;
oSettings.oPreviousSearch.sSearch=oData.sFilter;oSettings.aaSorting=oData.aaSorting.slice();
oSettings.saved_aaSorting=oData.aaSorting.slice();if(typeof oData.sFilterEsc!="undefined"){oSettings.oPreviousSearch.bEscapeRegex=oData.sFilterEsc
}if(typeof oData.aaSearchCols!="undefined"){for(var i=0;i<oData.aaSearchCols.length;
i++){oSettings.aoPreSearchCols[i]={sSearch:oData.aaSearchCols[i][0],bEscapeRegex:oData.aaSearchCols[i][1]}
}}if(typeof oData.abVisCols!="undefined"){oInit.saved_aoColumns=[];for(i=0;i<oData.abVisCols.length;
i++){oInit.saved_aoColumns[i]={};oInit.saved_aoColumns[i].bVisible=oData.abVisCols[i]
}}}}function _fnCreateCookie(sName,sValue,iSecs){var date=new Date();date.setTime(date.getTime()+(iSecs*1000));
sName+="_"+window.location.pathname.replace(/[\/:]/g,"").toLowerCase();document.cookie=sName+"="+encodeURIComponent(sValue)+"; expires="+date.toGMTString()+"; path=/"
}function _fnReadCookie(sName){var sNameEQ=sName+"_"+window.location.pathname.replace(/[\/:]/g,"").toLowerCase()+"=";
var sCookieContents=document.cookie.split(";");for(var i=0;i<sCookieContents.length;
i++){var c=sCookieContents[i];while(c.charAt(0)==" "){c=c.substring(1,c.length)}if(c.indexOf(sNameEQ)===0){return decodeURIComponent(c.substring(sNameEQ.length,c.length))
}}return null}function _fnGetUniqueThs(nThead){var nTrs=nThead.getElementsByTagName("tr");
if(nTrs.length==1){return nTrs[0].getElementsByTagName("th")}var aLayout=[],aReturn=[];
var ROWSPAN=2,COLSPAN=3,TDELEM=4;var i,j,k,iLen,jLen,iColumnShifted;var fnShiftCol=function(a,i,j){while(typeof a[i][j]!="undefined"){j++
}return j};var fnAddRow=function(i){if(typeof aLayout[i]=="undefined"){aLayout[i]=[]
}};for(i=0,iLen=nTrs.length;i<iLen;i++){fnAddRow(i);var iColumn=0;var nTds=[];for(j=0,jLen=nTrs[i].childNodes.length;
j<jLen;j++){if(nTrs[i].childNodes[j].nodeName=="TD"||nTrs[i].childNodes[j].nodeName=="TH"){nTds.push(nTrs[i].childNodes[j])
}}for(j=0,jLen=nTds.length;j<jLen;j++){var iColspan=nTds[j].getAttribute("colspan")*1;
var iRowspan=nTds[j].getAttribute("rowspan")*1;if(!iColspan||iColspan===0||iColspan===1){iColumnShifted=fnShiftCol(aLayout,i,iColumn);
aLayout[i][iColumnShifted]=(nTds[j].nodeName=="TD")?TDELEM:nTds[j];if(iRowspan||iRowspan===0||iRowspan===1){for(k=1;
k<iRowspan;k++){fnAddRow(i+k);aLayout[i+k][iColumnShifted]=ROWSPAN}}iColumn++}else{iColumnShifted=fnShiftCol(aLayout,i,iColumn);
for(k=0;k<iColspan;k++){aLayout[i][iColumnShifted+k]=COLSPAN}iColumn+=iColspan}}}for(i=0,iLen=aLayout[0].length;
i<iLen;i++){for(j=0,jLen=aLayout.length;j<jLen;j++){if(typeof aLayout[j][i]=="object"){aReturn.push(aLayout[j][i])
}}}return aReturn}function _fnMap(oRet,oSrc,sName,sMappedName){if(typeof sMappedName=="undefined"){sMappedName=sName
}if(typeof oSrc[sName]!="undefined"){oRet[sMappedName]=oSrc[sName]}}this.oApi._fnInitalise=_fnInitalise;
this.oApi._fnLanguageProcess=_fnLanguageProcess;this.oApi._fnAddColumn=_fnAddColumn;
this.oApi._fnAddData=_fnAddData;this.oApi._fnGatherData=_fnGatherData;this.oApi._fnDrawHead=_fnDrawHead;
this.oApi._fnDraw=_fnDraw;this.oApi._fnAjaxUpdate=_fnAjaxUpdate;this.oApi._fnAddOptionsHtml=_fnAddOptionsHtml;
this.oApi._fnFeatureHtmlFilter=_fnFeatureHtmlFilter;this.oApi._fnFeatureHtmlInfo=_fnFeatureHtmlInfo;
this.oApi._fnFeatureHtmlPaginate=_fnFeatureHtmlPaginate;this.oApi._fnPageChange=_fnPageChange;
this.oApi._fnFeatureHtmlLength=_fnFeatureHtmlLength;this.oApi._fnFeatureHtmlProcessing=_fnFeatureHtmlProcessing;
this.oApi._fnProcessingDisplay=_fnProcessingDisplay;this.oApi._fnFilterComplete=_fnFilterComplete;
this.oApi._fnFilterColumn=_fnFilterColumn;this.oApi._fnFilter=_fnFilter;this.oApi._fnSortingClasses=_fnSortingClasses;
this.oApi._fnVisibleToColumnIndex=_fnVisibleToColumnIndex;this.oApi._fnColumnIndexToVisible=_fnColumnIndexToVisible;
this.oApi._fnNodeToDataIndex=_fnNodeToDataIndex;this.oApi._fnVisbleColumns=_fnVisbleColumns;
this.oApi._fnBuildSearchArray=_fnBuildSearchArray;this.oApi._fnDataToSearch=_fnDataToSearch;
this.oApi._fnCalculateEnd=_fnCalculateEnd;this.oApi._fnConvertToWidth=_fnConvertToWidth;
this.oApi._fnCalculateColumnWidths=_fnCalculateColumnWidths;this.oApi._fnArrayCmp=_fnArrayCmp;
this.oApi._fnDetectType=_fnDetectType;this.oApi._fnGetDataMaster=_fnGetDataMaster;
this.oApi._fnGetTrNodes=_fnGetTrNodes;this.oApi._fnGetTdNodes=_fnGetTdNodes;this.oApi._fnEscapeRegex=_fnEscapeRegex;
this.oApi._fnReOrderIndex=_fnReOrderIndex;this.oApi._fnColumnOrdering=_fnColumnOrdering;
this.oApi._fnClearTable=_fnClearTable;this.oApi._fnSaveState=_fnSaveState;this.oApi._fnLoadState=_fnLoadState;
this.oApi._fnCreateCookie=_fnCreateCookie;this.oApi._fnReadCookie=_fnReadCookie;this.oApi._fnGetUniqueThs=_fnGetUniqueThs;
this.oApi._fnReDraw=_fnReDraw;var _that=this;return this.each(function(){var i=0,iLen,j,jLen;
for(i=0,iLen=_aoSettings.length;i<iLen;i++){if(_aoSettings[i].nTable==this){alert("DataTables warning: Unable to re-initialise DataTable. Please use the API to make any configuration changes required.");
return _aoSettings[i]}}var oSettings=new classSettings();_aoSettings.push(oSettings);
var bInitHandedOff=false;var bUsePassedData=false;var sId=this.getAttribute("id");
if(sId!==null){oSettings.sTableId=sId;oSettings.sInstance=sId}else{oSettings.sInstance=_oExt._oExternConfig.iNextUnique++
}oSettings.nTable=this;oSettings.oApi=_that.oApi;if(typeof oInit!="undefined"&&oInit!==null){_fnMap(oSettings.oFeatures,oInit,"bPaginate");
_fnMap(oSettings.oFeatures,oInit,"bLengthChange");_fnMap(oSettings.oFeatures,oInit,"bFilter");
_fnMap(oSettings.oFeatures,oInit,"bSort");_fnMap(oSettings.oFeatures,oInit,"bInfo");
_fnMap(oSettings.oFeatures,oInit,"bProcessing");_fnMap(oSettings.oFeatures,oInit,"bAutoWidth");
_fnMap(oSettings.oFeatures,oInit,"bSortClasses");_fnMap(oSettings.oFeatures,oInit,"bServerSide");
_fnMap(oSettings,oInit,"asStripClasses");_fnMap(oSettings,oInit,"fnRowCallback");
_fnMap(oSettings,oInit,"fnHeaderCallback");_fnMap(oSettings,oInit,"fnFooterCallback");
_fnMap(oSettings,oInit,"fnInitComplete");_fnMap(oSettings,oInit,"fnServerData");_fnMap(oSettings,oInit,"aaSorting");
_fnMap(oSettings,oInit,"aaSortingFixed");_fnMap(oSettings,oInit,"sPaginationType");
_fnMap(oSettings,oInit,"sAjaxSource");_fnMap(oSettings,oInit,"iCookieDuration");_fnMap(oSettings,oInit,"sDom");
_fnMap(oSettings,oInit,"oSearch","oPreviousSearch");_fnMap(oSettings,oInit,"aoSearchCols","aoPreSearchCols");
_fnMap(oSettings,oInit,"iDisplayLength","_iDisplayLength");_fnMap(oSettings,oInit,"bJQueryUI","bJUI");
if(typeof oInit.fnDrawCallback=="function"){oSettings.aoDrawCallback.push({fn:oInit.fnDrawCallback,sName:"user"})
}if(oSettings.oFeatures.bServerSide&&oSettings.oFeatures.bSort&&oSettings.oFeatures.bSortClasses){oSettings.aoDrawCallback.push({fn:_fnSortingClasses,sName:"server_side_sort_classes"})
}if(typeof oInit.bJQueryUI!="undefined"&&oInit.bJQueryUI){oSettings.oClasses=_oExt.oJUIClasses;
if(typeof oInit.sDom=="undefined"){oSettings.sDom='<"H"lfr>t<"F"ip>'}}if(typeof oInit.iDisplayStart!="undefined"&&typeof oSettings.iInitDisplayStart=="undefined"){oSettings.iInitDisplayStart=oInit.iDisplayStart;
oSettings._iDisplayStart=oInit.iDisplayStart}if(typeof oInit.bStateSave!="undefined"){oSettings.oFeatures.bStateSave=oInit.bStateSave;
_fnLoadState(oSettings,oInit);oSettings.aoDrawCallback.push({fn:_fnSaveState,sName:"state_save"})
}if(typeof oInit.aaData!="undefined"){bUsePassedData=true}if(typeof oInit!="undefined"&&typeof oInit.aoData!="undefined"){oInit.aoColumns=oInit.aoData
}if(typeof oInit.oLanguage!="undefined"){if(typeof oInit.oLanguage.sUrl!="undefined"&&oInit.oLanguage.sUrl!==""){oSettings.oLanguage.sUrl=oInit.oLanguage.sUrl;
$.getJSON(oSettings.oLanguage.sUrl,null,function(json){_fnLanguageProcess(oSettings,json,true)
});bInitHandedOff=true}else{_fnLanguageProcess(oSettings,oInit.oLanguage,false)}}}else{oInit={}
}if(typeof oInit.asStripClasses=="undefined"){oSettings.asStripClasses.push(oSettings.oClasses.sStripOdd);
oSettings.asStripClasses.push(oSettings.oClasses.sStripEven)}var nThead=this.getElementsByTagName("thead");
var nThs=nThead.length===0?null:_fnGetUniqueThs(nThead[0]);var bUseCols=typeof oInit.aoColumns!="undefined";
for(i=0,iLen=bUseCols?oInit.aoColumns.length:nThs.length;i<iLen;i++){var oCol=bUseCols?oInit.aoColumns[i]:null;
var nTh=nThs?nThs[i]:null;if(typeof oInit.saved_aoColumns!="undefined"&&oInit.saved_aoColumns.length==iLen){if(oCol===null){oCol={}
}oCol.bVisible=oInit.saved_aoColumns[i].bVisible}_fnAddColumn(oSettings,oCol,nTh)
}for(i=0,iLen=oSettings.aaSorting.length;i<iLen;i++){var oColumn=oSettings.aoColumns[oSettings.aaSorting[i][0]];
if(typeof oSettings.aaSorting[i][2]=="undefined"){oSettings.aaSorting[i][2]=0}if(typeof oInit.aaSorting=="undefined"&&typeof oSettings.saved_aaSorting=="undefined"){oSettings.aaSorting[i][1]=oColumn.asSorting[0]
}for(j=0,jLen=oColumn.asSorting.length;j<jLen;j++){if(oSettings.aaSorting[i][1]==oColumn.asSorting[j]){oSettings.aaSorting[i][2]=j;
break}}}if(this.getElementsByTagName("thead").length===0){this.appendChild(document.createElement("thead"))
}if(this.getElementsByTagName("tbody").length===0){this.appendChild(document.createElement("tbody"))
}if(bUsePassedData){for(i=0;i<oInit.aaData.length;i++){_fnAddData(oSettings,oInit.aaData[i])
}}else{_fnGatherData(oSettings)}oSettings.aiDisplay=oSettings.aiDisplayMaster.slice();
if(oSettings.oFeatures.bAutoWidth){_fnCalculateColumnWidths(oSettings)}oSettings.bInitialised=true;
if(bInitHandedOff===false){_fnInitalise(oSettings)}})}})(jQuery);
var bibtexify = (function($) {
    // helper function to "compile" LaTeX special characters to HTML
    var htmlify = function(str) {
        // TODO: this is probably not a complete list..
        str = str.replace(/\\"\{a\}/g, '&auml;')
            .replace(/\{\\aa\}/g, '&aring;')
            .replace(/\\aa\{\}/g, '&aring;')
            .replace(/\\"a/g, '&auml;')
            .replace(/\\"\{o\}/g, '&ouml;')
            .replace(/\\'e/g, '&eacute;')
            .replace(/\\'\{e\}/g, '&eacute;')
            .replace(/\\'a/g, '&aacute;')
            .replace(/\\'A/g, '&Aacute;')
            .replace(/\\"o/g, '&ouml;')
            .replace(/\\ss\{\}/g, '&szlig;')
            .replace(/\{/g, '')
            .replace(/\}/g, '')
            .replace(/\\&/g, '&')
            .replace(/--/g, '&ndash;');
        return str;
    };
    var uriencode = function(str) {
        // TODO: this is probably not a complete list..
        str = str.replace(/\\"\{a\}/g, '%C3%A4')
            .replace(/\{\\aa\}/g, '%C3%A5')
            .replace(/\\aa\{\}/g, '%C3%A5')
            .replace(/\\"a/g, '%C3%A4')
            .replace(/\\"\{o\}/g, '%C3%B6')
            .replace(/\\'e/g, '%C3%A9')
            .replace(/\\'\{e\}/g, '%C3%A9')
            .replace(/\\'a/g, '%C3%A1')
            .replace(/\\'A/g, '%C3%81')
            .replace(/\\"o/g, '%C3%B6')
            .replace(/\\ss\{\}/g, '%C3%9F')
            .replace(/\{/g, '')
            .replace(/\}/g, '')
            .replace(/\\&/g, '%26')
            .replace(/--/g, '%E2%80%93');
        return str;
    };
    // helper functions to turn a single bibtex entry into HTML
    var bib2html = {
        // the main function which turns the entry into HTML
        entry2html: function(entryData, bib) {
            var type = entryData.entryType.toLowerCase();
            // default to type misc if type is unknown
            if(array_keys(bib2html).indexOf(type) === -1) {
                type = 'misc';
                entryData.entryType = type;
            }
            var itemStr = htmlify(bib2html[type](entryData));
            itemStr += bib2html.links(entryData);
            itemStr += bib2html.bibtex(entryData);
            if (bib.options.tweet && entryData.url) {
                itemStr += bib2html.tweet(entryData, bib);
            }
            return itemStr.replace(/undefined/g,
                                   '<span class="undefined">missing<\/span>');
        },
        // converts the given author data into HTML
        authors2html: function(authorData) {
            var authorsStr = '';
            for (var index = 0; index < authorData.length; index++) {
                if (index > 0) { authorsStr += ", "; }
                authorsStr += authorData[index].last;
            }
            return htmlify(authorsStr);
        },
        // adds links to the PDF or url of the item
        links: function(entryData) {
            var itemStr = '';
            if (entryData.url && entryData.url.match(/.*\.pdf/)) {
                itemStr += ' (<a title="PDF-version of this article" href="' +
                            entryData.url + '">pdf<\/a>)';
            } else if (entryData.url) {
                itemStr += ' (<a title="This article online" href="' + entryData.url +
                            '">link<\/a>)';
            }
            return itemStr;
        },
        // adds the bibtex link and the opening div with bibtex content
        bibtex: function(entryData) {
            var itemStr = '';
            itemStr += ' (<a title="This article as BibTeX" id="cite_'+ entryData.cite  + '" href="#" class="biblink">' +
                        'bib</a>)<div class="bibinfo hidden">';
            itemStr += '<a href="#" class="bibclose" title="Close">x</a><pre>';
            itemStr += '@' + entryData.entryType + "{" + entryData.cite + ",\n";
            $.each(entryData, function(key, value) {
                if (key == 'author') {
                    itemStr += '  author = { ';
                    for (var index = 0; index < value.length; index++) {
                        if (index > 0) { itemStr += " and "; }
                        itemStr += value[index].last;
                    }
                    itemStr += ' },\n';
                } else if (key != 'entryType' && key != 'cite') {
                    itemStr += '  ' + key + " = { " + value + " },\n";
                }
            });
            itemStr += "}</pre></div>";
            return itemStr;
        },
        // generates the twitter link for the entry
        tweet: function(entryData, bib) {
          // url, via, text
          var itemStr = ' (<a title="Tweet this article" href="http://twitter.com/share?url=';
          itemStr += entryData.url;
          itemStr += '&via=' + bib.options.tweet;
          itemStr += '&text=';
          var splitName = function(wholeName) {
            var spl = wholeName.split(' ');
            return spl[spl.length-1];
          };
          var auth = entryData.author;
          if (auth.length == 1) {
            itemStr += uriencode(splitName(auth[0].last));
          } else if (auth.length == 2) {
            itemStr += uriencode(splitName(auth[0].last) + "%26" + splitName(auth[1].last));
          } else {
            itemStr += uriencode(splitName(auth[0].last) + " et al");
          }
          itemStr += ": " + encodeURIComponent('"' + entryData.title + '"');
          itemStr += '" target="_blank">tweet</a>)';
          return itemStr;
        },
        // helper functions for formatting different types of bibtex entries
        inproceedings: function(entryData) {
            return this.authors2html(entryData.author) + " (" + entryData.year + "). " +
                entryData.title + ". In <em>" + entryData.booktitle +
                ", pp. " + entryData.pages +
                ((entryData.address)?", " + entryData.address:"") + ".<\/em>";
        },
        article: function(entryData) {
            return this.authors2html(entryData.author) + " (" + entryData.year + "). " +
                entryData.title + ". <em>" + entryData.journal + ", " + entryData.volume +
                ((entryData.number)?"(" + entryData.number + ")":"")+ ", " +
                "pp. " + entryData.pages + ". " +
                ((entryData.address)?entryData.address + ".":"") + "<\/em>";
        },
        misc: function(entryData) {
            return this.authors2html(entryData.author) + " (" + entryData.year + "). " +
                entryData.title + ". " +
                ((entryData.howpublished)?entryData.howpublished + ". ":"") +
                ((entryData.note)?entryData.note + ".":"");
        },
        mastersthesis: function(entryData) {
            return this.authors2html(entryData.author) + " (" + entryData.year + "). " +
            entryData.title + ". " + entryData.type + ". " +
            entryData.organization + ", " + entryData.school + ".";
        },
        techreport: function(entryData) {
            return this.authors2html(entryData.author) + " (" + entryData.year + "). " +
                entryData.title + ". " + entryData.institution + ". " +
                entryData.number + ". " + entryData.type + ".";
        },
        book: function(entryData) {
            return this.authors2html(entryData.author) + " (" + entryData.year + "). " +
                " <em>" + entryData.title + "<\/em>, " +
                entryData.publisher + ", " + entryData.year +
                ((entryData.issn)?", ISBN: " + entryData.issn + ".":".");
        },
        inbook: function(entryData) {
            return this.authors2html(entryData.author) + " (" + entryData.year + "). " +
                entryData.chapter + " in <em>" + entryData.title + "<\/em>, " +
                ((entryData.editor)?" Edited by " + entryData.editor + ", ":"") +
                entryData.publisher + ", pp. " + entryData.pages + "" +
                ((entryData.series)?", <em>" + entryData.series + "<\/em>":"") +
                ((entryData.volume)?", Vol. " + entryData.volume + "":"") +
                ((entryData.issn)?", ISBN: " + entryData.issn + "":"") +
                ".";
        },
        // weights of the different types of entries; used when sorting
        importance: {
            'TITLE': 9999,
            'misc': 0,
            'manual': 10,
            'techreport': 20,
            'mastersthesis': 30,
            'inproceedings': 40,
            'incollection': 50,
            'proceedings': 60,
            'conference': 70,
            'article': 80,
            'phdthesis': 90,
            'inbook': 100,
            'book': 110,
            'unpublished': 120
        },
        // labels used for the different types of entries
        labels: {
            'article': 'Journal',
            'book': 'Book',
            'conference': 'Conference',
            'inbook': 'Book chapter',
            'incollection': '',
            'inproceedings': 'Conference',
            'manual': 'Manual',
            'mastersthesis': 'Thesis',
            'misc': 'Misc',
            'phdthesis': 'PhD Thesis',
            'proceedings': 'Conference proceeding',
            'techreport': 'Technical report',
            'unpublished': 'Unpublished'}
    };
    // format a phd thesis similarly to masters thesis
    bib2html.phdthesis = bib2html.mastersthesis;
    // conference is the same as inproceedings
    bib2html.conference = bib2html.inproceedings;

    // event handlers for the bibtex links
    var EventHandlers = {
        showbib: function showbib(event) {
            $(this).next(".bibinfo").removeClass('hidden').addClass("open");
            $("#shutter").show();
            event.preventDefault();
        },
        hidebib: function hidebib(event) {
            $("#shutter").hide();
            $(".bibinfo.open").removeClass("open").addClass("hidden");
            event.preventDefault();
        }
    };

    var Bib2HTML = function(data, $pubTable, options) {
        this.options = options;
        this.$pubTable = $pubTable;
        this.stats = { };
        this.initialize(data);
    };
    var bibproto = Bib2HTML.prototype;
    bibproto.initialize = function initialize(data) {
        var bibtex = new BibTex();
        bibtex.content = data;
        bibtex.parse();
        var bibentries = [], len = bibtex.data.length;
        var entryTypes = {};
        jQuery.extend(true, bib2html, this.options.bib2html);
        for (var index = 0; index < len; index++) {
            var item = bibtex.data[index];
            if (!item.year) {
              item.year = this.options.defaultYear || "To Appear";
            }
            var html = bib2html.entry2html(item, this);
            bibentries.push([item.year, bib2html.labels[item.entryType], html]);
            entryTypes[bib2html.labels[item.entryType]] = item.entryType;
            this.updateStats(item);
        }
        jQuery.fn.dataTableExt.oSort['type-sort-asc'] = function(x, y) {
            var item1 = bib2html.importance[entryTypes[x]],
                item2 = bib2html.importance[entryTypes[y]];
            return ((item1 < item2) ? -1 : ((item1 > item2) ?  1 : 0));
        };
        jQuery.fn.dataTableExt.oSort['type-sort-desc'] = function(x, y) {
            var item1 = bib2html.importance[entryTypes[x]],
                item2 = bib2html.importance[entryTypes[y]];
            return ((item1 < item2) ? 1 : ((item1 > item2) ?  -1 : 0));
        };
        var table = this.$pubTable.dataTable({ 'aaData': bibentries,
                              'aaSorting': this.options.sorting,
                              'aoColumns': [ { "sTitle": "Year" },
                                             { "sTitle": "Type", "sType": "type-sort", "asSorting": [ "desc", "asc" ] },
                                             { "sTitle": "Publication", "bSortable": false }],
                              'bPaginate': false
                            });
        if (this.options.visualization) {
            this.addBarChart();
        }
        $("th", this.$pubTable).unbind("click").click(function(e) {
          var $this = $(this),
              $thElems = $this.parent().find("th"),
              index = $thElems.index($this);
          if ($this.hasClass("sorting_disabled")) { return; }
          $this.toggleClass("sorting_asc").toggleClass("sorting_desc");

          if (index === 0) {
            table.fnSort( [[0, $thElems.eq(0).hasClass("sorting_asc")?"asc":"desc"],
                        [1, $thElems.eq(1).hasClass("sorting_asc")?"asc":"desc"]]);
          } else {
            table.fnSort( [[1, $thElems.eq(1).hasClass("sorting_asc")?"asc":"desc"],
                          [0, $thElems.eq(0).hasClass("sorting_asc")?"asc":"desc"]]);
          }
        });
        // attach the event handlers to the bib items
        $(".biblink", this.$pubTable).on('click', EventHandlers.showbib);
        $(".bibclose", this.$pubTable).on('click', EventHandlers.hidebib);
    };
    // updates the stats, called whenever a new bibtex entry is parsed
    bibproto.updateStats = function updateStats(item) {
        if (!this.stats[item.year]) {
            this.stats[item.year] = { 'count': 1, 'types': {} };
            this.stats[item.year].types[item.entryType] = 1;
        } else {
            this.stats[item.year].count += 1;
            if (this.stats[item.year].types[item.entryType]) {
                this.stats[item.year].types[item.entryType] += 1;
            } else {
                this.stats[item.year].types[item.entryType] = 1;
            }
        }
    };
    // adds the barchart of year and publication types
    bibproto.addBarChart = function addBarChart() {
        var yearstats = [], max = 0;
        $.each(this.stats, function(key, value) {
            max = Math.max(max, value.count);
            yearstats.push({'year': key, 'count': value.count,
                'item': value, 'types': value.types});
        });
        yearstats.sort(function(a, b) {
            var diff = a.year - b.year;
            if (!isNaN(diff)) {
              return diff;
            } else if (a.year < b.year) {
              return -1;
            } else if (a.year > b.year) {
              return 1;
            }
            return 0;
        });
        var chartIdSelector = "#" + this.$pubTable[0].id + "pubchart";
        var pubHeight = $(chartIdSelector).height()/max - 2;
        var styleStr = chartIdSelector +" .year { width: " +
                        (100.0/yearstats.length) + "%; }" +
                        chartIdSelector + " .pub { height: " + pubHeight + "px; }";
        var legendTypes = [];
        var stats2html = function(item) {
            var types = [],
                str = '<div class="year">',
                sum = 0;
            $.each(item.types, function(type, value) {
              types.push(type);
              sum += value;
            });
            types.sort(function(x, y) {
              return bib2html.importance[y] - bib2html.importance[x];
            });
            str += '<div class="filler" style="height:' + ((pubHeight+2)*(max-sum)) + 'px;"></div>';
            for (var i = 0; i < types.length; i++) {
                var type = types[i];
                if (legendTypes.indexOf(type) === -1) {
                    legendTypes.push(type);
                }
                for (var j = 0; j < item.types[type]; j++) {
                    str += '<div class="pub ' + type + '"></div>';
                }
            }
            return str + '<div class="yearlabel">' + item.year + '</div></div>';
        };
        var statsHtml = "<style>" + styleStr + "</style>";
        yearstats.forEach(function(item) {
            statsHtml += stats2html(item);
        });
        var legendHtml = '<div class="legend">';
        for (var i = 0, l = legendTypes.length; i < l; i++) {
            var legend = legendTypes[i];
            legendHtml += '<span class="pub ' + legend + '"></span>' + bib2html.labels[legend];
        }
        legendHtml += '</div>';
        $(chartIdSelector).html(statsHtml).after(legendHtml);
    };

    // Creates a new publication list to the HTML element with ID
    // bibElemId. The bibsrc can be
    //   - a jQuery selector, in which case html of the element is used
    //     as the bibtex data
    //   - a URL, which is loaded and used as data. Note, that same-origin
    //     policy restricts where you can load the data.
    // Supported options:
    //   - visualization: A boolean to control addition of the visualization.
    //                    Defaults to true.
    //   - tweet: Twitter username to add Tweet links to bib items with a url field.
    //   - sorting: Control the default sorting of the list. Defaults to [[0, "desc"],
    //              [1, "desc"]]. See http://datatables.net/api fnSort for details
    //              on formatting.
    //   - bib2html: Can be used to override any of the functions or properties of
    //               the bib2html object. See above, starting around line 40.
    return function(bibsrc, bibElemId, opts) {
        var options = $.extend({}, {'visualization': true,
                                'sorting': [[0, "asc"], [1, "desc"]]},
                                opts);
        var $pubTable = $("#" + bibElemId).addClass("bibtable");
        if ($("#shutter").size() === 0) {
            $pubTable.before('<div id="shutter" class="hidden"></div>');
            $("#shutter").click(EventHandlers.hidebib);
        }
        if (options.visualization) {
            $pubTable.before('<div id="' + bibElemId + 'pubchart" class="bibchart"></div>');
        }
        var $bibSrc = $(bibsrc);
        if ($bibSrc.length) { // we found an element, use its HTML as bibtex
            new Bib2HTML($bibSrc.html(), $pubTable, options);
            $bibSrc.hide();
        } else { // otherwise we assume it is a URL
            var callbackHandler = function(data) {
                new Bib2HTML(data, $pubTable, options);
            };
            $.get(bibsrc, callbackHandler, "text");
        }
    };
})(jQuery);

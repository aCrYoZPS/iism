use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::net::{Marking, Net, Transition};

pub struct ParseError {
    /// 1-based line number, 0 for errors about the text as a whole.
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            write!(f, "{}", self.message)
        } else {
            write!(f, "line {}: {}", self.line, self.message)
        }
    }
}

/// Places are numbered in order of first appearance in the text.
#[derive(Default)]
struct Places {
    names: Vec<String>,
    index: HashMap<String, usize>,
}

impl Places {
    fn id(&mut self, name: &str) -> usize {
        if let Some(&id) = self.index.get(name) {
            return id;
        }
        let id = self.names.len();
        self.names.push(name.to_string());
        self.index.insert(name.to_string(), id);
        id
    }
}

/// Parses a net description:
///
/// ```text
/// t1: p1 -> p2 p3      # transition: inputs -> outputs
/// marking: p1=1        # unlisted places hold 0 tokens
/// ```
///
/// Every error found is reported, not only the first one.
pub fn parse(text: &str) -> Result<(Net, Marking), Vec<ParseError>> {
    let mut errors = Vec::new();
    let mut places = Places::default();
    let mut transitions: Vec<Transition> = Vec::new();
    let mut names = HashSet::new();
    let mut marking_line: Option<(usize, &str)> = None;

    for (i, raw) in text.lines().enumerate() {
        let line_no = i + 1;
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("marking:") {
            if marking_line.is_some() {
                errors.push(error(line_no, "marking is given twice"));
            } else {
                marking_line = Some((line_no, rest));
            }
            continue;
        }
        match parse_transition(line, &mut places) {
            Ok(t) if !names.insert(t.name.clone()) => errors.push(error(
                line_no,
                format!("transition {} is defined twice", t.name),
            )),
            Ok(t) => transitions.push(t),
            Err(message) => errors.push(error(line_no, message)),
        }
    }

    let mut marking = vec![0; places.names.len()];
    if let Some((line_no, rest)) = marking_line {
        let mut seen = HashSet::new();
        for word in rest.split_whitespace() {
            match parse_tokens(word, &places.index) {
                Ok((p, _)) if !seen.insert(p) => errors.push(error(
                    line_no,
                    format!("place {} is given twice", places.names[p]),
                )),
                Ok((p, count)) => marking[p] = count,
                Err(message) => errors.push(error(line_no, message)),
            }
        }
    }

    if transitions.is_empty() && errors.is_empty() {
        errors.push(error(0, "the net has no transitions"));
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let net = Net {
        places: places.names,
        transitions,
    };
    Ok((net, marking))
}

fn error(line: usize, message: impl Into<String>) -> ParseError {
    ParseError {
        line,
        message: message.into(),
    }
}

fn parse_transition(
    line: &str,
    places: &mut Places,
) -> Result<Transition, String> {
    let (name, arcs) = line
        .split_once(':')
        .ok_or("expected `name: inputs -> outputs`")?;
    let name = name.trim();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Err("transition name must be one word".to_string());
    }
    let (inputs, outputs) =
        arcs.split_once("->").ok_or("missing `->` in transition")?;
    if outputs.contains("->") {
        return Err("only one `->` per transition".to_string());
    }
    Ok(Transition {
        name: name.to_string(),
        source: place_list(inputs, places)?,
        target: place_list(outputs, places)?,
    })
}

/// Duplicates would mean an arc of weight 2, which the net does not support.
fn place_list(text: &str, places: &mut Places) -> Result<Vec<usize>, String> {
    let mut list = Vec::new();
    for word in text.split_whitespace() {
        let id = places.id(word);
        if list.contains(&id) {
            return Err(format!(
                "place {word} is repeated, arcs of weight above 1 are not \
                 supported"
            ));
        }
        list.push(id);
    }
    Ok(list)
}

/// Parses a marking given on its own, e.g. `p1=1 p6=2`, for a known net.
/// Places that are not listed hold 0 tokens.
pub fn parse_marking(text: &str, net: &Net) -> Result<Marking, String> {
    let index: HashMap<String, usize> = net
        .places
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, name)| (name, i))
        .collect();
    let mut marking = vec![0; net.places.len()];
    let mut seen = HashSet::new();
    for word in text.split_whitespace() {
        let (p, count) = parse_tokens(word, &index)?;
        if !seen.insert(p) {
            return Err(format!("place {} is given twice", net.places[p]));
        }
        marking[p] = count;
    }
    Ok(marking)
}

fn parse_tokens(
    word: &str,
    index: &HashMap<String, usize>,
) -> Result<(usize, u32), String> {
    let (name, count) = word
        .split_once('=')
        .ok_or_else(|| format!("expected `place=count`, got `{word}`"))?;
    let &place = index
        .get(name)
        .ok_or_else(|| format!("unknown place {name} in marking"))?;
    let count = count
        .parse()
        .map_err(|_| format!("bad token count in `{word}`"))?;
    Ok((place, count))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::nets;

    type Described = (
        BTreeMap<String, u32>,
        BTreeMap<String, (BTreeSet<String>, BTreeSet<String>)>,
    );

    /// Place numbering depends on the text, so compare by names.
    fn describe(net: &Net, marking: &Marking) -> Described {
        let names = |ids: &[usize]| -> BTreeSet<String> {
            ids.iter().map(|&p| net.places[p].clone()).collect()
        };
        let tokens = net.places.iter().cloned().zip(marking.clone()).collect();
        let transitions = net
            .transitions
            .iter()
            .map(|t| (t.name.clone(), (names(&t.source), names(&t.target))))
            .collect();
        (tokens, transitions)
    }

    fn messages(text: &str) -> Vec<String> {
        match parse(text) {
            Ok(_) => Vec::new(),
            Err(errors) => errors.iter().map(|e| e.to_string()).collect(),
        }
    }

    #[test]
    fn lab_texts_match_hand_built_nets() {
        for (text, (net, m)) in [
            (nets::LEFT_TEXT, nets::left()),
            (nets::RIGHT_TEXT, nets::right()),
        ] {
            let (parsed, pm) = parse(text).ok().unwrap();
            assert_eq!(describe(&parsed, &pm), describe(&net, &m));
        }
    }

    #[test]
    fn comments_blank_lines_and_missing_marking() {
        let (net, m) = parse("# net\n\nt: p1 -> p2  # move\n").ok().unwrap();
        assert_eq!(net.places, vec!["p1", "p2"]);
        assert_eq!(m, vec![0, 0]);
    }

    #[test]
    fn transition_without_inputs_is_allowed() {
        let (net, _) = parse("t: -> p1").ok().unwrap();
        assert!(net.transitions[0].source.is_empty());
        assert_eq!(net.transitions[0].target, vec![0]);
    }

    #[test]
    fn repeated_place_is_rejected() {
        let e = messages("t: p1 p1 -> p2");
        assert_eq!(e.len(), 1);
        assert!(e[0].starts_with("line 1: place p1 is repeated"));
        assert_eq!(messages("t: p1 -> p2 p2").len(), 1);
    }

    #[test]
    fn same_place_in_input_and_output_is_allowed() {
        assert!(parse("t: p1 -> p1 p2").is_ok());
    }

    #[test]
    fn all_errors_are_collected_with_line_numbers() {
        let e = messages(
            "t1: p1 -> p2\nt1: p2 -> p1\nbroken\nt3: p1 p2\nmarking: p9=1 p1=x p1",
        );
        assert_eq!(e.len(), 6, "{e:?}");
        assert_eq!(e[0], "line 2: transition t1 is defined twice");
        assert!(e[1].starts_with("line 3:"));
        assert!(e[2].starts_with("line 4: missing `->`"));
        assert!(e[3].starts_with("line 5: unknown place p9"));
        assert!(e[4].starts_with("line 5: bad token count"));
        assert!(e[5].starts_with("line 5: expected `place=count`"));
    }

    #[test]
    fn marking_for_known_net() {
        let (net, _) = parse(nets::RIGHT_TEXT).ok().unwrap();
        // Places in order of appearance: p1 p2 p3 p6 p4 p5.
        assert_eq!(
            parse_marking("p6=2 p4=1", &net).unwrap(),
            vec![0, 0, 0, 2, 1, 0]
        );
        assert_eq!(parse_marking("", &net).unwrap(), vec![0; 6]);
        assert!(parse_marking("p9=1", &net).is_err());
        assert!(parse_marking("p1=1 p1=1", &net).is_err());
        assert!(parse_marking("p1", &net).is_err());
    }

    #[test]
    fn duplicate_marking_and_empty_text() {
        assert_eq!(
            messages("t: p1 -> p2\nmarking: p1=1 p1=2"),
            vec!["line 2: place p1 is given twice"]
        );
        assert_eq!(
            messages("t: p1 -> p2\nmarking: p1=1\nmarking: p2=1"),
            vec!["line 3: marking is given twice"]
        );
        assert_eq!(messages("# nothing"), vec!["the net has no transitions"]);
    }
}

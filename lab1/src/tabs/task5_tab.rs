use std::{collections::HashMap, fs, path::PathBuf};

use crate::common::{colors, tab::Tab};

use eframe::egui;
use rand::seq::SliceRandom;

#[derive(Debug, Clone)]
struct Team {
    name: String,
    rating: f64,

    eliminated: bool,
    current_place: Option<usize>,
    stage_reached: usize,
}

#[derive(Debug, Clone)]
struct MatchResult {
    left: String,
    right: String,
    winner: String,
    loser: String,
    left_rating: f64,
    right_rating: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TournamentState {
    NotStarted,
    InProgress,
    Finished,
}

#[derive(Debug)]
pub struct Task5Tab {
    teams: Vec<Team>,
    current_round: Vec<usize>,
    next_round: Vec<usize>,

    round_number: usize,
    total_rounds: usize,

    round_results: Vec<MatchResult>,
    history: Vec<MatchResult>,

    eliminated_by_round: Vec<Vec<usize>>,

    standings: Vec<usize>,

    state: TournamentState,
    load_error: Option<String>,
}

impl Task5Tab {
    pub fn new() -> Self {
        let mut tab = Self {
            teams: Vec::new(),
            current_round: Vec::new(),
            next_round: Vec::new(),

            round_number: 0,
            total_rounds: 0,

            round_results: Vec::new(),
            history: Vec::new(),

            eliminated_by_round: Vec::new(),
            standings: Vec::new(),

            state: TournamentState::NotStarted,
            load_error: None,
        };

        tab.load_teams();

        tab
    }

    fn teams_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("teams.toml")
    }

    fn load_teams(&mut self) {
        self.teams.clear();
        self.current_round.clear();
        self.next_round.clear();
        self.round_results.clear();
        self.history.clear();
        self.eliminated_by_round.clear();
        self.standings.clear();

        let path = Self::teams_path();

        let contents = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) => {
                self.load_error =
                    Some(format!("Could not read {}: {error}", path.display()));
                return;
            }
        };

        let entries: HashMap<String, f64> = match toml::from_str(&contents) {
            Ok(entries) => entries,
            Err(error) => {
                self.load_error = Some(format!(
                    "Could not parse {}: {error}",
                    path.display()
                ));
                return;
            }
        };

        let count = entries.len();

        if count == 0 {
            self.load_error = Some("The teams file is empty.".to_owned());
            return;
        }

        if count > 64 {
            self.load_error = Some(
                "The tournament supports at most 64 teams because k must be <= 6."
                    .to_owned(),
            );
            return;
        }

        if !count.is_power_of_two() {
            self.load_error = Some(format!(
                "The number of teams must be a power of two. Current count: {count}."
            ));
            return;
        }

        for (name, rating) in &entries {
            if name.trim().is_empty() {
                self.load_error =
                    Some("A team name cannot be empty.".to_owned());
                return;
            }

            if !rating.is_finite() || !(0.0..=100.0).contains(rating) {
                self.load_error = Some(format!(
                    "Rating for team {name:?} must be between 0 and 100."
                ));
                return;
            }
        }

        let mut teams: Vec<Team> = entries
            .into_iter()
            .map(|(name, rating)| Team {
                name,
                rating,
                eliminated: false,
                current_place: None,
                stage_reached: 0,
            })
            .collect();

        teams.sort_by(|a, b| a.name.cmp(&b.name));

        self.teams = teams;
        self.total_rounds = self.teams.len().trailing_zeros() as usize;
        self.load_error = None;

        self.reset_tournament();
    }

    fn reset_tournament(&mut self) {
        for team in &mut self.teams {
            team.eliminated = false;
            team.current_place = None;
            team.stage_reached = 0;
        }

        self.current_round = (0..self.teams.len()).collect();

        let mut rng = rand::rng();
        self.current_round.shuffle(&mut rng);

        self.next_round.clear();

        self.round_number = 0;

        self.round_results.clear();
        self.history.clear();

        self.eliminated_by_round.clear();
        self.standings.clear();

        self.state = TournamentState::NotStarted;
    }

    fn start_tournament(&mut self) {
        self.reset_tournament();

        if self.teams.len() == 1 {
            self.build_final_standings(0);
            self.state = TournamentState::Finished;
        }
    }

    fn winner_probability(left_rating: f64, right_rating: f64) -> f64 {
        let difference = left_rating - right_rating;
        1.0 / (1.0 + 10.0f64.powf(difference / 78.0))
    }

    fn choose_winner(left: &Team, right: &Team) -> bool {
        let left_probability =
            Self::winner_probability(left.rating, right.rating);
        rand::random::<f64>() < left_probability
    }

    fn build_final_standings(&mut self, champion_index: usize) {
        self.standings.clear();

        self.teams[champion_index].current_place = Some(1);
        self.standings.push(champion_index);

        let mut place = 2;

        for eliminated_in_round in self.eliminated_by_round.iter().rev() {
            for &team_index in eliminated_in_round {
                self.teams[team_index].current_place = Some(place);
                self.standings.push(team_index);
                place += 1;
            }
        }
    }

    fn make_step(&mut self) {
        if self.teams.is_empty() || self.state == TournamentState::Finished {
            return;
        }

        if self.state == TournamentState::NotStarted {
            self.start_tournament();

            if self.state == TournamentState::Finished {
                return;
            }
        }

        self.round_number += 1;

        self.round_results.clear();
        self.next_round.clear();

        let mut eliminated_this_round = Vec::new();

        for pair in self.current_round.chunks_exact(2) {
            let left_index = pair[0];
            let right_index = pair[1];

            let left = self.teams[left_index].clone();
            let right = self.teams[right_index].clone();

            let left_wins = Self::choose_winner(&left, &right);

            let (winner_index, loser_index) = if left_wins {
                (left_index, right_index)
            } else {
                (right_index, left_index)
            };

            self.teams[winner_index].stage_reached = self.round_number;

            self.teams[loser_index].eliminated = true;
            self.teams[loser_index].stage_reached = self.round_number;

            self.teams[loser_index].current_place = None;

            eliminated_this_round.push(loser_index);
            self.next_round.push(winner_index);

            self.round_results.push(MatchResult {
                left: left.name,
                right: right.name,
                winner: self.teams[winner_index].name.clone(),
                loser: self.teams[loser_index].name.clone(),
                left_rating: left.rating,
                right_rating: right.rating,
            });
        }

        self.history.extend(self.round_results.clone());
        self.eliminated_by_round.push(eliminated_this_round);

        self.current_round = std::mem::take(&mut self.next_round);

        if self.current_round.len() == 1 {
            let champion_index = self.current_round[0];

            self.build_final_standings(champion_index);

            self.state = TournamentState::Finished;
        } else {
            self.state = TournamentState::InProgress;
        }
    }

    fn state_label(&self) -> &'static str {
        match self.state {
            TournamentState::NotStarted => "Not started",
            TournamentState::InProgress => "In progress",
            TournamentState::Finished => "Finished",
        }
    }

    fn stage_label(&self) -> String {
        if self.state == TournamentState::Finished {
            return "Final".to_owned();
        }

        if self.round_number == 0 {
            return format!("Round of {}", self.teams.len());
        }

        match self.current_round.len() {
            2 => "Final".to_owned(),
            4 => "Semifinal".to_owned(),
            8 => "Quarterfinal".to_owned(),
            n => format!("Round of {n}"),
        }
    }

    fn winner_index(&self) -> Option<usize> {
        if self.state != TournamentState::Finished {
            return None;
        }

        self.teams
            .iter()
            .position(|team| team.current_place == Some(1))
    }

    fn draw_team_info(&self, ui: &mut egui::Ui) {
        ui.heading("Teams");

        egui::Grid::new("teams_grid")
            .striped(true)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.label("Team");
                ui.label("Rating");
                ui.label("Status");
                ui.label("Place");
                ui.label("Stage");
                ui.end_row();

                let mut indices: Vec<usize> = (0..self.teams.len()).collect();

                indices.sort_by(|&left, &right| {
                    let left_place =
                        self.teams[left].current_place.unwrap_or(usize::MAX);
                    let right_place =
                        self.teams[right].current_place.unwrap_or(usize::MAX);

                    left_place.cmp(&right_place).then_with(|| {
                        self.teams[right]
                            .rating
                            .partial_cmp(&self.teams[left].rating)
                            .unwrap()
                    })
                });

                for index in indices {
                    let team = &self.teams[index];

                    ui.label(&team.name);
                    ui.label(format!("{:.0}", team.rating));

                    if team.current_place == Some(1) {
                        ui.label(
                            egui::RichText::new("Winner").color(colors::GREEN),
                        );
                    } else if team.eliminated {
                        ui.label(
                            egui::RichText::new("Eliminated")
                                .color(colors::RED),
                        );
                    } else {
                        ui.label("Playing");
                    }

                    ui.label(
                        team.current_place
                            .map(|place| place.to_string())
                            .unwrap_or_else(|| "-".to_owned()),
                    );

                    let stage = if team.stage_reached == 0 {
                        "Not played".to_owned()
                    } else {
                        format!("Round {}", team.stage_reached)
                    };

                    ui.label(stage);

                    ui.end_row();
                }
            });
    }

    fn draw_standings(&self, ui: &mut egui::Ui) {
        ui.heading("Standings");

        if self.state != TournamentState::Finished {
            ui.label(
                egui::RichText::new(
                    "Final standings will be available after the final match.",
                )
                .weak(),
            );
            return;
        }

        egui::Grid::new("standings_grid")
            .striped(true)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.label("Place");
                ui.label("Team");
                ui.label("Rating");
                ui.end_row();

                for (position, &team_index) in self.standings.iter().enumerate()
                {
                    let team = &self.teams[team_index];

                    let label_text: String;
                    if position <= 1 {
                        label_text = (position + 1).to_string()
                    } else {
                        let lower_equivalent = 2u32.pow((position).ilog2()) + 1;
                        let higher_equivalent =
                            2u32.pow((position).ilog2() + 1);
                        label_text =
                            format!("{lower_equivalent}-{higher_equivalent}");
                    }
                    ui.label(label_text);

                    if position == 0 {
                        ui.label(
                            egui::RichText::new(&team.name)
                                .color(colors::GREEN),
                        );
                    } else {
                        ui.label(&team.name);
                    }

                    ui.label(format!("{:.0}", team.rating));
                    ui.end_row();
                }
            });
    }

    fn draw_round_results(&self, ui: &mut egui::Ui) {
        ui.heading("Latest matches");

        if self.round_results.is_empty() {
            ui.label(
                egui::RichText::new("Press \u{201c}Make step\u{201d} to simulate the next stage.")
                    .weak(),
            );
            return;
        }

        egui::Grid::new("round_results_grid")
            .striped(true)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.label("Match");
                ui.label("Ratings");
                ui.label("Winner");
                ui.label("Eliminated");
                ui.end_row();

                for result in &self.round_results {
                    ui.label(format!("{} vs {}", result.left, result.right));
                    ui.label(format!(
                        "{:.0} : {:.0}",
                        result.left_rating, result.right_rating
                    ));
                    ui.label(
                        egui::RichText::new(&result.winner)
                            .color(colors::GREEN),
                    );
                    ui.label(
                        egui::RichText::new(&result.loser).color(colors::RED),
                    );
                    ui.end_row();
                }
            });
    }
}

impl Tab for Task5Tab {
    fn name(&self) -> &'static str {
        "Task5"
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Football Cup Simulator");

        ui.add_space(8.0);

        if let Some(error) = &self.load_error {
            ui.colored_label(colors::RED, error);
            return;
        }

        ui.horizontal(|ui| {
            ui.label(format!("Stage: {}", self.stage_label()));
            ui.separator();
            ui.label(format!("Status: {}", self.state_label()));
            ui.separator();
            ui.label(format!("Teams: {}", self.teams.len()));
            ui.separator();
            ui.label(format!(
                "Round: {}/{}",
                self.round_number, self.total_rounds
            ));
        });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui.button("Reset").clicked() {
                self.reset_tournament();
            }

            let can_make_step = self.state != TournamentState::Finished;

            if ui
                .add_enabled(can_make_step, egui::Button::new("Make step"))
                .clicked()
            {
                self.make_step();
            }

            if let Some(winner_index) = self.winner_index() {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!(
                        "Winner: {}",
                        self.teams[winner_index].name
                    ))
                    .color(colors::GREEN),
                );
            }
        });

        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.draw_round_results(ui);
                ui.add_space(16.0);
                self.draw_standings(ui);
                ui.add_space(16.0);
                self.draw_team_info(ui);
            });
    }
}

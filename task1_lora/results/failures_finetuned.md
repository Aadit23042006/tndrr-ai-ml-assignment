# Failures for `finetuned` (7/100 not exact)

## test-076  [shifted]
**Request:** I need a code review at quarter past twelve in the afternoon on 2027-03-08 for an hour and a quarter in Rooftop Cafe with Tanvi. Please confirm.

- gold: `{"title": "code review", "date": "2027-03-08", "time": "12:15", "duration_minutes": 75, "location": "Rooftop Cafe", "attendees": ["Tanvi"]}`
- pred: `{"title": "code review", "date": "2027-03-08", "time": "12:15", "duration_minutes": 120, "location": "Rooftop Cafe", "attendees": ["Tanvi"]}`
- wrong fields: duration_minutes

## test-078  [shifted]
**Request:** I need a town hall on 20-Nov-2026 at quarter past five in the evening for an hour and a quarter.

- gold: `{"title": "town hall", "date": "2026-11-20", "time": "17:15", "duration_minutes": 75, "location": null, "attendees": []}`
- pred: `{"title": "town hall", "date": "2026-11-20", "time": "17:15", "duration_minutes": 90, "location": null, "attendees": []}`
- wrong fields: duration_minutes

## test-080  [shifted]
**Request:** Can you block time for a doctor visit at quarter past ten in the morning on the 27th of May 2026 for 1 hour 45 minutes. Please confirm.

- gold: `{"title": "doctor visit", "date": "2026-05-27", "time": "10:15", "duration_minutes": 105, "location": null, "attendees": []}`
- pred: `{"title": "doctor visit", "date": "2026-05-27", "time": "10:15", "duration_minutes": 145, "location": null, "attendees": []}`
- wrong fields: duration_minutes

## test-084  [shifted]
**Request:** Please arrange a sprint planning at quarter past seven in the morning on the 17th of October 2026 lasting 40 minutes and invite Kavya over Teams.

- gold: `{"title": "sprint planning", "date": "2026-10-17", "time": "07:15", "duration_minutes": 40, "location": "Teams", "attendees": ["Kavya"]}`
- pred: `{"title": "sprint planning", "date": "2026-10-17", "time": "07:15", "duration_minutes": 40, "location": null, "attendees": ["Kavya"]}`
- wrong fields: location

## test-085  [shifted]
**Request:** Please arrange an one on one at quarter past six in the evening on the 12th of October 2027 with Tanvi for 1 hour 45 minutes. Thanks!

- gold: `{"title": "one on one", "date": "2027-10-12", "time": "18:15", "duration_minutes": 105, "location": null, "attendees": ["Tanvi"]}`
- pred: `{"title": "one on one", "date": "2027-10-12", "time": "18:15", "duration_minutes": 180, "location": null, "attendees": ["Tanvi"]}`
- wrong fields: duration_minutes

## test-089  [shifted]
**Request:** Can you block time for a doctor visit at quarter past eleven in the morning on the 1st of May 2026 with Sanjay in Conference Room B for 1 hour 45 minutes. Thanks!

- gold: `{"title": "doctor visit", "date": "2026-05-01", "time": "11:15", "duration_minutes": 105, "location": "Conference Room B", "attendees": ["Sanjay"]}`
- pred: `{"title": "doctor visit", "date": "2026-05-01", "time": "11:15", "duration_minutes": 60, "location": "Conference Room B", "attendees": ["Sanjay"]}`
- wrong fields: duration_minutes

## test-099  [shifted]
**Request:** Please arrange a dentist appointment at 08:00 on October 16th, 2027 for 1 hour 45 minutes on Teams.

- gold: `{"title": "dentist appointment", "date": "2027-10-16", "time": "08:00", "duration_minutes": 105, "location": "Teams", "attendees": []}`
- pred: `{"title": "dentist appointment", "date": "2027-10-16", "time": "08:00", "duration_minutes": 180, "location": "Teams", "attendees": []}`
- wrong fields: duration_minutes

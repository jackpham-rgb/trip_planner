# Current Setup

## What exists today

## The workbook

The current system is a single Excel file, Jack Master Travel Database, built and maintained with Python and openpyxl. It is the third file being sent alongside this one, so treat it as the actual source of data, not just a description of it.

It is organized into 19 tabs. Eight are master reference sheets that apply across every region: a read me and legend, a Lists sheet that feeds every dropdown in the workbook, an Events Calendar of date sensitive events worldwide, a Viral and Social Media sheet rating social media famous places honestly, a Trip Comparison sheet that scores every trip version on eight dimensions with a live overall score and rank, a Pacing Rules sheet, a Local versus Tourist sheet, and a Country Notes sheet for visas, entry rules, and safety.

The remaining tabs are the actual destinations, grouped by region: Cali (California only, organized by trip length and area rather than country), Weekend Trips (one to three day getaways, newly added), USA, USA MEX CAN (Canada and Mexico, including cross border trips), South America, Europe, East Asia, South East Asia, Middle East, Africa, and Rest of World. Long trip regions generally offer four versions each: Solo two week, Solo one month, Partner two week, and Partner one month, built on two philosophies. Solo trips aim for density, seeing as much as possible. Partner trips are redesigned from scratch to cut anything redundant and spend the saved time actually enjoying a place, rather than being the solo version with items removed.

Every itinerary row follows the same 27 column shape: version, day, date, country or state, city or region, overnight location, route, transport, transit time, main destination, specific attraction or activity, address, website, why it is worth it, time needed, priority (must, high, or optional), type, time of day, cost, whether it needs a reservation or a guide, whether it is seasonal or weather dependent, an honest viral rating, a food or drink pick, an information class (fact, current, seasonal, estimate, personal recommendation, or verify), and a best time or recommended date. Priority, viral rating, and information class all have color coding and dropdowns wired to the Lists sheet.

## What this workbook is not, yet

It is a reference, not a running system. There is no tracking of what Jack has actually done, no repeat or do not repeat flag, no ranking that adapts to his behavior, and no way to ask it a quick question and get a short answer back. Every use of it today means opening the file and reading.

## Prior planning toward an app

Before this recommendation platform idea came up, Jack and Claude had already started scoping a plan to turn this workbook into a personal Android app, and that plan is still the closest thing to prior art for this new project.

The plan was Kotlin with Jetpack Compose for the app itself, with two separate SQLite databases managed through Room: one holding the content (pulled from the workbook, replaced whenever the workbook updates) and one holding state (what Jack has checked off, his notes, never touched by a content update). Keeping those two databases strictly separate was identified as the key design decision, so that refreshing the destination data can never silently wipe out personal history.

One blocking issue was flagged and not yet resolved: itinerary rows in the workbook do not have stable identifiers, so if content changes, checkmarks and notes tied to a row could silently attach to the wrong row afterward. The proposed fix was to add a structured ID to every row, in a form like JP-S2W-D06-03 for a Japan, Solo two week, day six, row three entry, plus latitude and longitude columns, before the workbook grows any larger.

The five features prioritized for a first version of that app were a today view, checking items off, a queue of everything still marked verify, filtering by trip version, and tapping a location to open it directly in Google Maps through a geo link.

None of this Android specific work has been built yet. It is documented here because the new recommendation platform idea should either build on this plan, replace it with something better, or explain why a different direction is preferred, rather than starting from a blank page and accidentally redoing the same analysis.

## Tools already in use

Python and openpyxl for building and maintaining the workbook, with data content kept in separate files from the assembly script that writes the actual Excel file. Every version of the workbook is recalculated and checked before being sent back to Jack, so formulas and dropdowns are confirmed working, not just assumed to work.

# Smart Davomat AI — Real Backend MVP

## What this backend does
- Browser camera capture
- Real face enrollment using `face_recognition`
- Real face matching against enrolled students
- Attendance record creation
- Confidence score
- YOLO cell-phone detection on the same frame
- SQLite attendance database
- `/api/health`, `/api/enroll`, `/api/recognize`, `/api/phone`, `/api/attendance`

## Important deployment note
This backend is designed for Render Docker deployment. Render supports Docker web services and public `onrender.com` URLs. See the official Render documentation.

For a persistent production database, move SQLite to Postgres or attach a persistent disk. For a short MVP demo, SQLite is sufficient.

## How to use
1. Open the public backend URL.
2. Start Camera.
3. Enter a student's name and click Enroll Current Face.
4. Click Recognize & Mark Attendance.
5. The backend performs real face matching and checks the frame for a cell phone with YOLO.

Do not upload private Google service-account credentials to GitHub.

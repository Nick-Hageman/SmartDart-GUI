import firebase_admin
from firebase_admin import credentials, db
import threading

class FirebaseClient:
    def __init__(self, game_id):
        cred = credentials.Certificate("serviceAccountKey.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                #'databaseURL': 'https://smartdart-9c426-default-rtdb.firebaseio.com/'
                'databaseURL': 'https://smartdart-97b28-default-rtdb.firebaseio.com/'
            })
        self.game_ref = db.reference(f'games/{game_id}')
        self.listeners = {}

        # Initialize game state here
        self.initialize_game_state()

    def initialize_game_state(self):
        self.game_ref.update({
            "turn": "player1",
            "turnNum": 0
        })

    def add_score(self, player, dart_values):
        player_ref = self.game_ref.child(player)
        scores = player_ref.get() or []
        if isinstance(scores, dict):
            scores = [scores[k] for k in sorted(scores.keys(), key=lambda x: int(x) if x.isdigit() else x)]
        scores.append(dart_values)
        player_ref.set(scores)

    def stream_scores(self, player, callback):
        def listener(event):
            if event.path == '/':
                callback(player, event.data)
            else:
                full_data = self.game_ref.child(player).get()
                callback(player, full_data)

        ref = self.game_ref.child(player)
        self.listeners[player] = ref.listen(listener)

    def stop_all_streams(self):
        for listener in self.listeners.values():
            listener.close()

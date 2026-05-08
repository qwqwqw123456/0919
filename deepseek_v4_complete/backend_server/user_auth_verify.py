class UserAuth:
    def verify(self, token):
        return token == "valid_token"

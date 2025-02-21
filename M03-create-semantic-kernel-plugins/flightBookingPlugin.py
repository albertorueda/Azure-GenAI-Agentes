from typing import TypedDict, Annotated
from typing import List
import json
import os
from semantic_kernel.functions import kernel_function

class FlightModel(TypedDict):
    id: int
    airline: str
    destination: str
    departure_date: str
    price: float
    is_booked: bool

    def __init__(self, id: int, airline: str, destination: str, departure_date: str, price: float, is_booked: bool = False):
        self.id = id
        self.airline = airline
        self.destination = destination
        self.departure_date = departure_date
        self.price = price
        self.is_booked = is_booked

class FlightBookingPlugin:
    
    def __init__(self):
        self.file_path = "flights.json"
        self.flights = self.load_flights_from_file()
    
    @kernel_function
    async def search_flights(self, destination: str, departure_date: str) -> List[FlightModel]:
        """Searches for available flights based on the destination and departure date in the format YYYY-MM-DD"""
        return [flight for flight in self.flights if flight["destination"].lower() == destination.lower() and flight["departure_date"] == departure_date]

    @kernel_function
    async def book_flight(self, flight_id: int) -> str:
        """Books a flight based on the flight ID provided"""
        flight = next((flight for flight in self.flights if flight["id"] == flight_id), None)
        if not flight:
            return "Flight not found. Please provide a valid flight ID."
        
        if flight["is_booked"]:
            return "You've already booked this flight."
        
        flight["is_booked"] = True
        self.save_flights_to_file()
        
        return f"Flight booked successfully! Airline: {flight['airline']}, Destination: {flight['destination']}, Departure: {flight['departure_date']}, Price: ${flight['price']}."
    
    def save_flights_to_file(self):
        with open(self.file_path, "w") as file:
            json.dump(self.flights, file, indent=4)
    
    def load_flights_from_file(self) -> List[FlightModel]:
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                return json.load(file)
        raise FileNotFoundError(f"The file '{self.file_path}' was not found. Please provide a valid flights.json file.")
